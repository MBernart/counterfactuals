from explainers_lib.explainers.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.remote import RemoteExplainerWorkerFactory

explainer = GrowingSpheresExplainer(step_size=0.1, max_radius=5.0, num_samples=1000)

from twisted.internet import reactor
reactor.listenTCP(8000, RemoteExplainerWorkerFactory(explainer))
reactor.run()
