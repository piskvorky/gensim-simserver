import gensim
from simserver import SessionServer

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

documents = [gensim.utils.simple_preprocess(doc) for doc in documents]
corpus = [{'id': '%i' % (num),
            'tokens': document}
            for num, document in enumerate(documents)]

server = SessionServer('/tmp/my_server')
server.train(corpus, method='lsi') # create a semantic model
server.index(corpus) # convert plain text to semantic representation and index it

print server.find_similar('1') # convert query to semantic representation and compare against index
server.delete(['1','2','3','4','5','6','7','8','0']) # incremental deleting also works
server.index(corpus)
print server.find_similar('1')
