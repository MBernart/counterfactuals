from explainers_lib.explainers.wachter import WachterExplainer
from explainers_lib.explainers.remote import RemoteExplainerWorkerFactory

explainer = WachterExplainer()

from twisted.internet import reactor

reactor.listenTCP(8000, RemoteExplainerWorkerFactory(explainer))
reactor.run()
