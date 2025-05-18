
from typing import Sequence
from twisted.internet import reactor, protocol, threads
from twisted.protocols.basic import LineReceiver
import pickle

from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset, SerializableDataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model, SerializableModel, TorchModel

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

import pickle
from twisted.internet import reactor, protocol, defer
from twisted.protocols.basic import LineReceiver
from twisted.python.threadpool import ThreadPool
from threading import Thread, Event
from twisted.internet import threads

class RemoteExplainerProtocol(LineReceiver):
    def __init__(self):
        self.buffer = b''
        self.expected_length = 0
        self.state = "COMMAND"
        self.timeout = 30  # seconds
        self.timeout_call = None
        self.train_deferred = None
        self.explain_deferred = None

    def connectionMade(self):
        print(f"Connected to worker at {self.transport.getPeer()}")

    def sendData(self, data):
        self.sendLine(b"DATA")
        self.sendLine(str(len(data)).encode())
        self.transport.write(data)

    def sendModel(self, model_data):
        self.sendLine(b"MODEL")
        self.sendLine(str(len(model_data)).encode())
        self.transport.write(model_data)

    def sendTrain(self):
        self.sendLine(b"TRAIN")
        self.train_deferred = defer.Deferred()
        return self.train_deferred

    def sendExplain(self):
        self.sendLine(b"EXPLAIN")
        self.explain_deferred = defer.Deferred()
        self.startTimeout()
        return self.explain_deferred

    def startTimeout(self):
        self.timeout_call = reactor.callLater(
            self.timeout, 
            self.handleTimeout,
            self.transport.getPeer()
        )

    def handleTimeout(self, worker_addr):
        print(f"Timeout occurred for worker {worker_addr}")
        if self.explain_deferred:
            self.explain_deferred.errback(TimeoutError("Explanation timed out"))
            self.explain_deferred = None
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
            elif self.state == "ERROR_LENGTH":
                self.handleError(payload)
            
            self.buffer = remaining
            self.state = "COMMAND"
            self.current_op = None
            self.setLineMode()
            if remaining:
                self.dataReceived(remaining)

    def handleResult(self, data):
        try:
            if data == b'TRAIN_OK':
                if self.train_deferred:
                    self.train_deferred.callback(True)
                    self.train_deferred = None
            else:
                result = pickle.loads(data)
                if self.explain_deferred:
                    self.explain_deferred.callback(result)
                    self.explain_deferred = None
                    if self.timeout_call:
                        self.timeout_call.cancel()
        except Exception as e:
            if self.explain_deferred:
                self.explain_deferred.errback(e)
                self.explain_deferred = None

    def handleError(self, data):
        error_msg = data.decode()
        print(f"Error from worker: {error_msg}")
        if self.train_deferred:
            self.train_deferred.errback(RuntimeError(error_msg))
            self.train_deferred = None
        if self.explain_deferred:
            self.explain_deferred.errback(RuntimeError(error_msg))
            self.explain_deferred = None
        self.transport.loseConnection()

class RemoteExplainerFactory(protocol.ClientFactory):
    def __init__(self):
        self.protocol = None
        self.connected = defer.Deferred()

    def buildProtocol(self, addr):
        self.protocol = RemoteExplainerProtocol()
        self.connected.callback(self.protocol)
        return self.protocol

    def clientConnectionFailed(self, connector, reason):
        self.connected.errback(reason)
        reactor.stop()

class RemoteExplainer(Explainer):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.factory = RemoteExplainerFactory()
        self.protocol = None
        self._reactor_thread = None
        self.connected_event = Event()
        self._start_reactor()

    def _start_reactor(self):
        if not reactor.running:
            self._reactor_thread = Thread(target=reactor.run, args=(False,))
            self._reactor_thread.daemon = True
            self._reactor_thread.start()
        reactor.callFromThread(self._connect)

    def _connect(self):
        connector = reactor.connectTCP(self.host, self.port, self.factory)
        self.factory.connected.addCallbacks(self._set_protocol, self._connection_failed)

    def _set_protocol(self, protocol):
        self.protocol = protocol
        self.connected_event.set()  # Signal that we're connected

    def _connection_failed(self, reason):
        print(f"Connection failed: {reason}")
        reactor.stop()

    def _ensure_connected(self):
        if not self.connected_event.wait(10):  # Wait up to 10 seconds for connection
            raise ConnectionError("Failed to connect to server")

    def sendData(self, data: SerializableDataset) -> None:
        self._ensure_connected()
        serialized = data.serialize()
        return threads.blockingCallFromThread(
            reactor,
            lambda: self.protocol.sendData(serialized)
        )

    def sendModel(self, model: SerializableModel) -> None:
        self._ensure_connected()
        serialized = model.serialize()
        return threads.blockingCallFromThread(
            reactor,
            lambda: self.protocol.sendModel(serialized)
        )

    def fit(self, model: SerializableModel, data: SerializableDataset) -> None:
        self._ensure_connected()
        self.sendData(data)
        self.sendModel(model)
        return threads.blockingCallFromThread(
            reactor,
            lambda: self.protocol.sendTrain()
        )

    def explain(self, model: SerializableModel, data: Dataset) -> Sequence[Counterfactual]:
        self._ensure_connected()
        # TODO: figure out when should we send the model and the data
        self.sendData(data)
        return threads.blockingCallFromThread(
            reactor,
            lambda: self.protocol.sendExplain()
        )