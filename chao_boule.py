import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation, rc
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\FFMPEG\bin\ffmpeg.exe'
import numba

fig = plt.figure(figsize=(10, 10))          #dimension fenêtre d'animation
ax = plt.gca()

ax.set_aspect('equal', adjustable='box')
plt.axis('off')
fig.patch.set_facecolor('grey')
R = 5 #rayon sphère

circle1 = plt.Circle((0, 0), R + 0.2, color='lightgrey', ec = "black", lw = 10, zorder = -1)    #cercle
ax.add_patch(circle1)

pas = 0.00002
t = np.arange(0,20,pas)

@numba.jit
def pos(x0, y0, xv0, yv0, t, R = R):

    """Fonction prenant les positions et vitesses initiales et renvoient une liste de
    coordonnées en x et une autre en y"""

    g = 9.81
    k = 1       #coefficient de restitution

    #initialisation
    v_pos = np.zeros((t.shape[0], 2))
    v_pos[0] = [x0, y0]

    v_vit = np.zeros((t.shape[0], 2))
    v_vit[0] = [xv0, yv0]

    #boucle calculant la position de la balle pour tout temps t
    for i in range(0, t.shape[0]-4):

        v_pos[i+1] = [ v_vit[i][0] * pas + v_pos[i][0],
            -1/2 * g * pas**2 + v_vit[i][1]*pas + v_pos[i][1]] #[x, y]

        #si la balle rencontre une surface
        if v_pos[i+1][0]**2 + v_pos[i+1][1]**2>=(R)**2 :

            a = -v_pos[i][0]/np.sqrt(R**2 - v_pos[i][0]**2) #pente de la tangente à la surface

            if v_pos[i+1][1]> 0 :
                a = -a

            vect = np.array([a, 1])
            v_vit[i+1][0], v_vit[i+1][1] = k * (v_vit[i] - 2 * np.dot(v_vit[i], vect)/(1+a**2) * vect)  #réflexion du vecteur vitesse

        else :
            v_vit[i+1] = [v_vit[i][0], -g * pas + v_vit[i][1]]

    return np.transpose(v_pos)[0], np.transpose(v_pos)[1]

colore = ["red", "maroon", "darkorange", "moccasin", "yellow", "green", "aquamarine", "teal", "violet"] #couleurs des balles
xxx = np.linspace(-0.5,-0.5001, 9)  #liste des positions intiales en x des balles
masse = []

#boucle qui calcule la trajectoire de chaque balle + initialise leur caractéristique pour l'animation
for i in range(9):

    x1, y1 = pos(xxx[i],0, 0, 0, t)
    #x1,y1 = pos(0,-4.95, 18, 0, t)

    masse1, = ax.plot([], [],"o",color = colore[i], markersize = 30)                #caractéristique de la balle
    trace1, = ax.plot([], [],".",color = colore[i], markersize = 0.5, zorder = 0)   #caractéristique de la trainée suivant la balle

    masse.append([x1,y1, masse1, trace1])

def animate(nn):

    """Animation : fonctionne comme une boucle while et maj l'écran à chaque boucle suivant les commandes imposées"""

    global masse

    n = 1000 * nn   #je prends une image toutes les mille fois pour éviter d'avoir une grosse animation

    d = n - 50000   #longueur de la trainée
    if n < 50000:
        d = 0

    #boucle qui va permette de maj chaque balle
    for i in range(9):
        masse[i][2].set_data([masse[i][0][n], masse[i][1][n]])              #set_data permet de maj la position d'un objet
        masse[i][3].set_data([masse[i][0][d:n:100], masse[i][1][d:n:100]])

    #return masse[0][2], masse[0][3], masse[1][2], masse[1][3], masse[2][2], masse[2][3]

    return masse[0][2], masse[0][3], masse[1][2], masse[1][3], masse[2][2], masse[2][3],\
    masse[3][2], masse[3][3], masse[4][2], masse[4][3], masse[5][2], masse[5][3], masse[6][2], masse[6][3],\
    masse[7][2], masse[7][3], masse[8][2], masse[8][3]
    #on return les objets qu'on veut maj


anim = animation.FuncAnimation(fig, animate, frames = int((t.shape[0]-1)/1000),
                               interval = 1, blit = True)                   #commande d'animation avec frames le nombre d'image total

#writervideo = animation.FFMpegWriter(fps=50)
#anim.save('acc.mp4',writer=writervideo)
plt.show()