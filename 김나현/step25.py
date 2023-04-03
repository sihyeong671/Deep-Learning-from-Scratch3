import graphviz
#from dezero.core_simple import Variable
from dezero import Variable

#x=Variable(np.array(1.0))
#print(x)

with open("sample.dot") as f:
    g=f.read()
t=graphviz.Source(g)
t.format='png'
#t.filename = 'sample'
t.directory = './'
#t.render(view=True)
t.view()
#pylab.savefig("sample.png")
#plt.show()