# # # #
# ReseauMLPclass.py
# @author Zhibin.LU
# @created Sept 14 2017 13:11:38 GMT-0500 (EST)
# @description 
# # # #


import numpy as np
import random
import time


def onehots(Y, n):
    '''n: combien de classes. Retourner une matrice dont chaque rangee est un onehot'''
    targets = np.array(Y, dtype=int).reshape(-1)
    one_hots = np.eye(n)[targets]
    return one_hots

def softmax(Xs):
    '''pour la couche de sortie. Implementation numerique stable de softmax'''
    Bs=np.max(Xs, axis=0)
    return np.exp(Xs-Bs)/np.sum(np.exp(Xs-Bs), axis=0)

def rect(Xs):
    '''
    definir la fonction de ReLu/Ramp/Rectifieur et sa dérivee'''
    return np.maximum(0.0, Xs)
    
def rect_derivate(Ys):
    '''
    definir la fonction de ReLu/Ramp/Rectifieur et sa derivee'''
    return np.where(Ys > 0.0, 1.0, 0.0)

def rect_prime(Ys):
    '''
    définir la fonction de ReLu/Ramp/Rectifieur et sa prime'''
    return np.where(Ys > 0.0, 1.0, 0.0)

class ReseauMLP(object):
    
    def __init__(self, sizes, activationFunction, outputFunction):
        '''
        # le reseau neurone plus general, realise par matrice calcule
        # sizes = [2, 3, 2]=[couche entree, couche cache, couche sortie]
        # activationFunction: la fonction d'activation
        # outputFunction: la fonction sortie
        '''
        
        # combien de couches, avec couche entree et couche sortie.
        self.num_layers = len(sizes)
        self.num_sortie_neurone = sizes[-1]
        self.sizes = sizes
        self.activationf = activationFunction
        self.outputf = outputFunction
        
        # On utilise vecteur colonne pour exprimer les biais de chaque couche 
        self.biases = [np.zeros(l)[:,None] for l in sizes[1:]] 
        # sample the weights of a layer from a uniform distribution in -1/√nc , 1/√nc 
        # , where nc is the number of inputs for this layer
        self.weights = [np.random.uniform(-l**-0.5,l**-0.5,lplus*l).reshape(lplus,l)
                        for lplus, l in zip(sizes[1:], sizes[:-1])]


    def compute_predictions(self, Xs):
        '''
        # prediction de forward propagation pour matrice echantillon.
        # Xs(row) n'inclut y, chaque rangee est un echantillon de d dimension.
        # retourner les valeurs de chaque neurone de sortie, par rangee(row)
        '''
        
        # HSs= sorties de chaque neurone de chaque couche, 
        # chaque vecteur vient de chaque echantillon
        HSs=Xs.T
        l=1
        for w, b in zip(self.weights, self.biases):
            if l < self.num_layers-1 :
                HSs = self.activationf(np.dot(w, HSs) + b)
                l+=1
            else:
                HSs = self.outputf(np.dot(w, HSs) + b)
        return HSs.T

    def evaluateAccuracy(self, test_data):
        '''
        # afficher le nombre de prediction correct pour test_data 
        # quand on fait chaque descente de gradient.
        '''
        
        if self.num_sortie_neurone>1:
            classes_pred=np.argmax(self.compute_predictions(test_data[:,:-1]), axis=1)
        else:
            # seulement pour output function=sigmoid
            classes_pred=np.where(self.compute_predictions(test_data[:,:-1])>0.5,1.0,0.0)
            classes_pred=classes_pred.reshape(len(classes_pred))
        correct= test_data[:,-1]==classes_pred
        return classes_pred[correct].size

    def evaluate(self, test_data):
        '''
        # test_data: row, inclut y
        # return taux d'erreur,num d'erreur, loss pour test_data 
        # quand on fait chaque descente de gradient.
        # retourne taux d'erreur, numero de erreur, loss
        '''
        
        if self.num_sortie_neurone>1:
            classes_pred=np.argmax(self.compute_predictions(test_data[:,:-1]), axis=1)
        else:
            # seulement pour output function=sigmoid
            classes_pred=np.where(self.compute_predictions(test_data[:,:-1])>0.5,1.0,0.0)
            classes_pred=classes_pred.reshape(len(classes_pred))
        err= test_data[:,-1]!=classes_pred
        n_err=classes_pred[err].size
        taux=float(n_err)/float(len(test_data))
        return ( taux, n_err, self.loss(test_data) )
       
            
    def fprop(self, Xs):
        '''
        # forward_propagation pour ha et hs=activationF(ha) pour chaque layer
        # obtenir toutes les valeurs avant les neurones et toutes les values 
        # apres les neurones pour tous les echantillons
        # Xs(col) n'inclut y
        '''
        
        # HSs= sorties de chaque neurone de chaque couche, 
        # chaque vecteur vient de chaque echantillon
        # HAs= w*hs^(l-1)+b , valeurs avant activationF, 
        # chaque vecteur vient de chaque echantillon
        HSs = Xs
        # obtenir toutes les valeurs avant les neurones et toutes les valeurs 
        # apres les neurones pour tous les echantillons
        HAs_layers = [ HSs ]
        HSs_layers = [ HSs ]
        l=1
        for w, b in zip(self.weights, self.biases):
            # Obtenir HAs de l eme couche
            HAs=np.dot(w, HSs) + b
            if l < self.num_layers-1 :
                HSs = self.activationf(HAs)
                l+=1
            else:
                # si c'est la dernier couche, utilise la function softmax.
                HSs = self.outputf(HAs)
            # HAs_layers et HSs_layers contiennent toutes les valeurs de neurones avant 
            # et apres la fonction d'activation. ca vient des echantillons de minibatch, 
            # chaque couche est une matrice dont chaque colonne correspond a un echantillon
            HAs_layers.append(HAs)
            HSs_layers.append(HSs)
            
        return (HAs_layers,HSs_layers)
    
    
    def bprop(self, mini_data, mu, lambda1, lambda2): 
        '''
        # backward propagation
        # retro-propagation le reseau en utilisant mini_data(echantillons), 
        # et mise a jour des poids et des biais
        # mini_data(row) inclut y
        # mu：taux d'apprentissage
        # lambda1 pour le  risque empirique r´egularis´e lambda1*||w||_1 pour chaque couche
        # lambda2 pour le  risque empirique r´egularis´e lambda2*||w||^2_2 pour chaque couche
        '''
        
        k=len(mini_data) # important ici, evite les problemes
        
        # pretraiter y en utilisant onehot
        if self.num_sortie_neurone>1:
            Ys=onehots( mini_data[:,-1] , self.num_sortie_neurone ).T
        else:
            Ys=mini_data[:,-1]
        mini_data=mini_data[:,:-1].T
        
        # obtenir toutes les valeurs avant les neurones et toutes les valeurs 
        # apres les neurones pour tous les echantillons
        HAs_layers,HSs_layers=self.fprop(mini_data)
        
        # obtenir les gradients pour les poids et biais
        Grad_Ws = [  ]
        Grad_Bs = [  ]
        # obtenir les gradients pour les neurones
        # la couche de sortie.
        Grad_HAs = [ HSs_layers[-1] - Ys ]
        Grad_HSs = [ np.zeros(self.num_sortie_neurone) ] #Grad_HS
        # faire la retro-propagation a partir de la 2eme couche a l'inverse
        # "l" est l'indice de couche
        for l in xrange( self.num_layers-2, -1, -1 ) : 
            # lambda1 pour le  risque empirique regularise lambda1*||w||_1 pour chaque couche
            # lambda2 pour le  risque empirique regularise lambda2*||w||^2_2 pour chaque couche
            regularizedGrad = lambda1*np.sign(self.weights[l]) + 2*lambda2*self.weights[l]
            # c'est deja le sum de minibatch(np.dot) de grad_W, 
            # apres on va diviser par len(minibatch)
            # dimension de grad_ws[l]=nombre de couche(l+1) * nombre de couche(l)
            # Grad_HAs[0] est la prochaine couche (l+1). 
            Grad_Ws.insert(0, np.dot(Grad_HAs[0], HSs_layers[l].T) + regularizedGrad*k ) 
            # grad_b = grad_ha de la prochaine couche, sum K echantillons de minibatch
            Grad_Bs.insert(0, np.sum(Grad_HAs[0], axis=1)[:,None] )
            # Il y a k colonne pour HSs[l], car ca vient par K echantillons de minibatch.
            # chaque vecteur vertical = dimension de HS,ca vient de une echantillons.
            Grad_HSs.insert(0, np.dot(self.weights[l].T, Grad_HAs[0]) )
            # obtenir la valeur de la derivation de la fonction d'activation
            derivation=eval(self.activationf.__name__+'_prime')(HAs_layers[l])
            # derivation=eval(self.activationf.__name__+'_derivate')(HSs_layers[l])
            Grad_HAs.insert(0, Grad_HSs[0] * derivation )

        # Descente du gradient pour chaque w,b cette fois de mini_batch
        self.weights = [w - gw * mu / k  for w, gw in zip(self.weights, Grad_Ws)]
        self.biases = [b - gb * mu / k  for b, gb in zip(self.biases, Grad_Bs)]

        
    def loss(self, Xs):
        '''
        # calculer la valeur de la perte avec des thetas(w,s) courants du reseau
        # en utilisant la function J(theta)=-log(Os_Yeme(x))
        # Xs(row) inclut y
        '''
        
        k = len(Xs)
        Ys=np.array(Xs[:,-1], dtype=int)
        Xs=Xs[:,:-1].T
        #obtenir les valeurs de neurones de la sortie en donnant les x de l'échantillon
        HAs_layers,HSs_layers=self.fprop(Xs)
        Os=HSs_layers[-1]
        # calculer la perte en utilisant la function J(theta)=-log(Os_Yeme(x))
        if self.num_sortie_neurone>1:
            perte = np.array([-np.log(o[y]) for o,y in zip(Os.T,Ys)]).sum() / k
        else:
            Os=Os.reshape(-1)
            # quand la sortie est seulement un neurone
            perte = ( -np.dot(Ys, np.log(Os).T) - np.dot((1 - Ys), np.log(1 - Os).T) ) / k

        return perte

    def gradiantDescentMiniBatchVite(self, training_data, epoque, K_batch_size, mu,
            lambda1, lambda2, validation_data=None, test_data=None, display=False):
        '''
        # la methode de descente du gradient pour minibatch
        # training_data(row*n): l'ensemble de donnees d'entrainement
        # epoque: la fois d'entrainement
        # K_batch_szie: la taille de data pour chaque fois d'entrainement
        # mu: taux d'apprentissage
        # lambda: hyper-parmetres pour le risque empirique regularis´e
        # lambda1 pour le  risque empirique regularise lambda1*||w||_1 pour chaque couche
        # lambda2 pour le  risque empirique regularise lambda2*||w||^2_2 pour chaque couche
        # validation_data(row*n): l'ensemble de donnees de validation
        # test_data(row*n): l'ensemble de donnees de test
        '''
        
        n = len(training_data)
        logs={}
        ELoss=[]
        ETaux=[]
        VLoss=[]
        VTaux=[]
        TLoss=[]
        TTaux=[]
        for ep in xrange(epoque):
            # diviser training data selon la taille de mini_batch
            mini_batches = [
                training_data[ k: k+K_batch_size ]
                for k in xrange(0, n, K_batch_size)]
            for i, mini_data in enumerate(mini_batches):
                # mise à jour  w et b un fois selon un petit ensemble d'entrainement de donnees
                self.bprop(mini_data, mu, lambda1, lambda2)
                
            # faire la validation
            if validation_data is not None:
                # pour plus rapide , calcule taux etc sur train data et validation data apres chaque epoque
                taux,err,loss=self.evaluate(training_data)
                ELoss.append(loss)
                ETaux.append(taux)
                if display:
                    print "Epoque {0} fini: {1} erreurs sur {2} train data, Taux d'erreur: {3}, Loss: {4}".format(
                        ep, err , len(training_data), taux, loss )
                taux,err,loss=self.evaluate(validation_data)
                VLoss.append(loss)
                VTaux.append(taux)
                if display:
                    print "Epoque {0} fini: {1} erreurs sur {2} validation data, Taux d'erreur: {3}, Loss: {4}".format(
                        ep, err , len(validation_data), taux, loss )
                    
            # faire le test
            if test_data is not None:
                taux,err,loss=self.evaluate(test_data)
                TLoss.append(loss)
                TTaux.append(taux)
                if display:
                    print "Epoque {0} fini: {1} erreurs sur {2} test data, Taux d'erreur: {3}, Loss: {4}".format(
                        ep, err , len(test_data), taux, loss )
                if taux < float(0.019) :
                    break
                    
            logs['ELoss']=ELoss
            logs['ETaux']=ETaux
            logs['VLoss']=VLoss
            logs['VTaux']=VTaux
            logs['TLoss']=TLoss
            logs['TTaux']=TTaux
        return logs



#test
reseauSizes=[784, 50, 10]
mu = 0.2
epoque=20
K_minibatch=30
lambda1=0.0
lambda2=0.0

print '\nReseauMLP Entrainement commence....'
# Faire l'entrainement
t1 = time.clock()
rmlp = ReseauMLP(reseauSizes,rect,softmax) 
logs5=rmlp.gradiantDescentMiniBatchVite(mnist_train_data, epoque, K_minibatch, mu, lambda1, lambda2,mnist_valid_data,mnist_test_data,display=True)
t2 = time.clock()
print 'Ca nous a pris ', t2-t1, ' secondes pour entrainer le reseau',rmlp.sizes,' sur ', mnist_train_data.shape[0],' points de training data .'
print 'taux, num err, loss sur validation=',rmlp.evaluate(mnist_valid_data)
print 'taux, num err, loss sur test=',rmlp.evaluate(mnist_test_data)
    