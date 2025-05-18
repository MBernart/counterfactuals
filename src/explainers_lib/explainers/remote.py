
from twisted.internet import reactor, protocol, threads
from twisted.protocols.basic import LineReceiver
import pickle

from explainers_lib.datasets import SerializableDataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import TorchModel

class RemoteExplainerWorkerProtocol(LineReceiver):
    def __init__(self, explainer: Explainer):
        self.dataset = None
        self.model = None
        self.state = "COMMAND"
        self.current_op = None
        self.expected_length = 0
        self.buffer = b''
        self.explainer = explainer

    def connectionMade(self):
        print(f"Main server connected from {self.transport.getPeer()}")

    def lineReceived(self, line):
        if self.state == "COMMAND":
            cmd = line.decode().strip()
            if cmd.startswith("DATA"):
                self.current_op = "DATA"
                self.state = "LENGTH"
            elif cmd.startswith("MODEL"):
                self.current_op = "MODEL"
                self.state = "LENGTH"
            elif cmd == "TRAIN":
                self.startTraining()
            elif cmd == "EXPLAIN":
                self.startExplaining()
            elif cmd == "SHUTDOWN":
                reactor.stop()
        elif self.state == "LENGTH":
            self.expected_length = int(line.decode().strip())
            self.state = "DATA"
            self.setRawMode()

    def rawDataReceived(self, data):
        self.buffer += data
        if len(self.buffer) >= self.expected_length:
            payload = self.buffer[:self.expected_length]
            remaining = self.buffer[self.expected_length:]
            
            if self.current_op == "DATA":
                self.dataset = SerializableDataset.deserialize(payload)
                print("Received dataset")
            elif self.current_op == "MODEL":
                self.model = TorchModel.deserialize(payload) # TODO: introduce ModelFactory
                print("Received model")
            
            self.buffer = remaining
            self.state = "COMMAND"
            self.current_op = None
            self.setLineMode()
            if remaining:
                self.dataReceived(remaining)

    def startTraining(self):
        if self.dataset and self.model:
            d = threads.deferToThread(self.trainModel)
            d.addCallback(lambda _: self.sendResult(b"TRAIN_OK"))
            d.addErrback(lambda f: self.sendError(f"Training failed: {f}"))

    def trainModel(self):
        print("Training explainer...")
        self.explainer.fit(self.model, self.dataset)
        return True

    def startExplaining(self):
        if self.dataset and self.model:
            d = threads.deferToThread(self.explain)
            d.addCallback(lambda res: self.sendResult(pickle.dumps(res)))
            d.addErrback(lambda f: self.sendError(f"Explanation failed: {f}"))

    def explain(self):
        print("Generating explainations...")
        return self.explainer.explain(self.model, self.dataset)

    def sendResult(self, data):
        self.sendLine(b"RESULT")
        self.sendLine(str(len(data)).encode())
        self.transport.write(data)

    def sendError(self, message):
        self.sendLine(b"ERROR")
        self.sendLine(str(len(message)).encode())
        self.transport.write(message.encode())

class RemoteExplainerWorkerFactory(protocol.Factory):
    def __init__(self, explainer: Explainer):
        self.explainer = explainer

    def buildProtocol(self, addr):
        return RemoteExplainerWorkerProtocol(self.explainer)

class RemoteExplainerProtocol(LineReceiver):
    def __init__(self, dataset: SerializableDataset, model):
        self.dataset = dataset
        self.model = model
        self.results = []
        self.operations = []
        self.timeout = 30  # seconds
        self.buffer = b''
        self.expected_length = 0
        self.current_op = None
        self.state = "COMMAND"

    def connectionMade(self):
        print(f"Connected to worker at {self.transport.getPeer()}")
        self.initOperationQueue()

    def initOperationQueue(self):
        self.operations = iter([
            self.sendDataset,
            self.sendModel,
            self.sendTrainCommand,
            self.sendExplainCommand
        ])
        reactor.callLater(0, self.nextOperation)

    def nextOperation(self):
        try:
            op = next(self.operations)
            op()
        except StopIteration:
            pass

    def sendDataset(self):
        data = self.dataset.serialize()
        self.sendLine(b"DATA")
        self.sendLine(str(len(data)).encode())
        self.transport.write(data)
        reactor.callLater(1, self.nextOperation)

    def sendModel(self):
        model_data = self.model.serialize()
        self.sendLine(b"MODEL")
        self.sendLine(str(len(model_data)).encode())
        self.transport.write(model_data)
        reactor.callLater(1, self.nextOperation)

    def sendTrainCommand(self):
        self.sendLine(b"TRAIN")
        reactor.callLater(1, self.nextOperation)

    def sendExplainCommand(self):
        self.sendLine(b"EXPLAIN")
        self.startTimeout()

    def startTimeout(self):
        self.timeout_call = reactor.callLater(
            self.timeout, 
            self.handleTimeout,
            self.transport.getPeer()
        )

    def handleTimeout(self, worker_addr):
        print(f"Timeout occurred for worker {worker_addr}")
        self.transport.loseConnection()

    def lineReceived(self, line):
        cmd = line.decode().strip()
        if cmd == "RESULT":
            self.current_op = "RESULT"
            self.state = "LENGTH"
        elif cmd == "ERROR":
            self.state = "ERROR_LENGTH"
        elif self.state == "LENGTH" or self.state == "ERROR_LENGTH":
            self.expected_length = int(line.decode().strip())
            self.state = "DATA"
            self.setRawMode()

    def rawDataReceived(self, data):
        self.buffer += data
        if len(self.buffer) >= self.expected_length:
            payload = self.buffer[:self.expected_length]
            remaining = self.buffer[self.expected_length:]
            
            if self.current_op == "RESULT":
                self.handleResult(payload)
            
            self.buffer = remaining
            self.state = "COMMAND"
            self.current_op = None
            self.setLineMode()
            if remaining:
                self.dataReceived(remaining)

    def handleResult(self, data):
        try:
            if data == b'TRAIN_OK':
                print("Explainer trained successfully")
            else:
                result = pickle.loads(data)
                self.results.append(result)
                print(f"Received result: {result}")
                self.timeout_call.cancel()
                self.transport.loseConnection()
        except Exception as e:
            print(f"Error processing result: {str(e)}")

    def handleError(self, data):
        error_msg = data.decode()
        print(f"Error from worker: {error_msg}")
        self.timeout_call.cancel()
        self.transport.loseConnection()

class RemoteExplainerFactory(protocol.ClientFactory):
    def __init__(self, dataset: SerializableDataset, model):
        self.dataset = dataset
        self.model = model

    def buildProtocol(self, addr):
        return RemoteExplainerProtocol(self.dataset, self.model)

    def clientConnectionFailed(self, connector, reason):
        print(f"Connection failed: {reason.getErrorMessage()}")
        reactor.stop()
