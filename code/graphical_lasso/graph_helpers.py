import numpy as np
import matplotlib.pyplot as plt


def dessin_graphe(P,Vx,Vy,f='black') :
    """ représente graphiquement un graphe dont P la matrice de precision est donné en argument
    Vx np.array des abscisses et Vy np.array des ordonnées des sommets """

    P = np.clip(P,-1,1)
    for i in range(P.shape[0]):
        for j in range(i) :
            col = (1,1,1)
            if P[i,j] != 0:
                if  P[i,j] >0:
                    col = tuple((1-P[i,j])*np.array((1,1,1))+P[i,j]*np.array((1,0,0)))
                else:
                    col = tuple((1+P[i,j])*np.array((1,1,1))-P[i,j]*np.array((0,0,1)))
                plt.plot([Vx[i],Vx[j]],[Vy[i],Vy[j]],c = col)

    plt.scatter(Vx,Vy,c = f,zorder = 100)

    plt.axis('off')
    plt.axis('equal')
    plt.show()


def init_graph(nom,n,k = 3,sigma=0.5,kappa=0.6):
    """ renvoie la matrice de précision d'un matrice de forme mot et de n sommets et le dessine ? """
    np.random.seed(0)
    if nom == "ligne" :
        A = np.eye(n, n, 1)+np.eye(n, n, -1)
        Vx = np.linspace(-n//2,n//2,n)
        Vy = np.zeros(n)
    elif nom == "cyclique":
        A = np.eye(n, n, 1)+np.eye(n, n, -1)
        A[n-1,0] = np.random.randn()
        A[0,n-1] = np.random.randn()

        V = np.array([np.exp(x*1j) for x in np.linspace(0,2*np.pi,n+1)])
        Vx = V.real
        Vy = V.imag
    elif nom == "grilleCarree":
        A = np.zeros((n,n))
        m =int(np.sqrt(n))
        for i in range(0,m):
            for j in range(0,m):
                if i>0:
                    A[i+j*m,(i-1)+j*m] = 1
                if j>0:
                    A[i+j*m,i+(j-1)*m] = 1
                if i<m-1:
                    A[i+j*m,(i+1)+j*m] = 1
                if j<m-1:
                    A[i+j*m,i+(j+1)*m] = 1
        a= np.linspace(0,1,m)
        Vx = np.hstack((a,)*m)
        Vy = np.vstack((a,)*m).flatten('F')

    elif nom == "kvoisins":
        Vx = np.random.rand(n,1)
        Vy = np.random.rand(n,1)
        distance =[[((Vx[i] - Vx[j])**2 + (Vy[i] - Vy[j])**2)**0.5 for j in range(0,n)] for i in range(0,n)]
        A = np.zeros((n,n))
        for i in range(0,n):
            l = distance[i]
            ll = sorted(list(enumerate(l)), key=lambda x:x[1])
            for j in ll[1:k+1]:
                A[i][j[0]] = 1
                A[j[0]][i] = 1
    elif nom == "rbf_random":
        vertices = np.random.random(size=(n, 2))
        Vx = vertices[:,0]
        Vy = vertices[:,1]
        A = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i+1, n):
                dist = np.linalg.norm(vertices[i]-vertices[j])
                if dist < kappa:
                    A[j, i] = A[i, j] = np.exp(-dist**2/(2*sigma**2))
    else :
        print("Ce nom de graphe n'est pas connu, choisir ligne/cyclique/grilleCarree/kvoisins/rbf_random")
        A,Vx,Vy = [],[],[]

    return A,Vx,Vy
