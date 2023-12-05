import cv2
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
import gc
from more_itertools import distinct_permutations as idp

class stimuli_gen:

    def __init__(self,searchtype,tang,dang,tcol,dcol,length=6,width=3,n224=4,N=12):
        self.target_angle    = tang
        self.dist_angle      = dang
        self.target_luminance    = tcol
        self.dist_luminance      = dcol
        self.searchtype      = searchtype
        self.imgsize         = n224
        self.N               = N
        self.width           = width
        self.length          = length

    def get_centre(self,i):
        centre_x=self.imgsize*112-np.round(self.imgsize*112*3/4*np.cos((i/self.N)*2*np.pi))
        centre_y=self.imgsize*112+np.round(self.imgsize*112*3/4*np.sin((i/self.N)*2*np.pi))
        return centre_x, centre_y 

    def draw_slope(self,image_to_draw_on,angle,i,contrast):
        centre_x,centre_y=self.get_centre(i)
        y = np.array(([int(centre_x+self.imgsize*self.length*np.sin(angle*np.pi/180)), int(centre_y-self.imgsize*self.length*np.cos(angle*np.pi/180))])) 
        x = np.array(([int(centre_x-self.imgsize*self.length*np.sin(angle*np.pi/180)), int(centre_y+self.imgsize*self.length*np.cos(angle*np.pi/180))])) 
        image_to_draw_on=cv2.line(image_to_draw_on,x,y,contrast*255,self.imgsize*self.width)
        return image_to_draw_on

    def multistimcreatorpres(self,setsize,trials):
        alltrials_cuelocs=[]
        for i in range(trials):
            alltrials_cuelocs.append(random.sample(range(1,13), setsize))
        alltrials_cuelocs=np.array(alltrials_cuelocs)
        all_stim=[]
        for i in alltrials_cuelocs:
            image_to_draw_on=128*np.ones((self.imgsize*224,self.imgsize*224))
            
            #DrawTarget
            image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,i[0],self.target_luminance)
            #DrawDists
            if self.searchtype=='conj':
                for m in i[1:]:
                    if random.randint(0, 1)==1:
                        image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,m,self.dist_luminance)
                    else:
                        image_to_draw_on=self.draw_slope(image_to_draw_on,self.dist_angle,m,self.target_luminance)
            
            if self.searchtype=='luminance':
                for m in i[1:]:
                    image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,m,self.dist_luminance) 
            

            if self.searchtype=='angle':
                for m in i[1:]:
                    image_to_draw_on=self.draw_slope(image_to_draw_on,self.dist_angle,m,self.target_luminance) 
            
            all_stim.append(image_to_draw_on)
        return np.array(all_stim)
    

    def multistimcreatorabs(self,setsize,trials):
        alltrials_cuelocs=[]
        for i in range(trials):
            alltrials_cuelocs.append(random.sample(range(1,13), setsize))
        alltrials_cuelocs=np.array(alltrials_cuelocs)
        all_stim=[]
        for i in alltrials_cuelocs:
            image_to_draw_on=128*np.ones((self.imgsize*224,self.imgsize*224))
            
            #DrawDists
            if self.searchtype=='conj':
                for m in i:
                    if random.randint(0, 1)==1:
                        image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,m,self.dist_luminance)
                    else:
                        image_to_draw_on=self.draw_slope(image_to_draw_on,self.dist_angle,m,self.target_luminance)
                        
            if self.searchtype=='luminance':
                for m in i:
                    image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,m,self.dist_luminance) 
            

            if self.searchtype=='angle':
                for m in i:
                    image_to_draw_on=self.draw_slope(image_to_draw_on,self.dist_angle,m,self.target_luminance) 
            
            all_stim.append(image_to_draw_on)
        return np.array(all_stim)

    def trainsetmaker(self,n_images,noise):
        set_present=np.concatenate((self.multistimcreatorpres(4,n_images),self.multistimcreatorpres(6,n_images),self.multistimcreatorpres(12,n_images)))
        set_absent=np.concatenate((self.multistimcreatorabs(4,n_images),self.multistimcreatorabs(6,n_images),self.multistimcreatorabs(12,n_images)))
        full_set=np.concatenate((set_present,set_absent))
        del set_present
        del set_absent
        gc.collect()
        full_set=(full_set+np.random.normal(0,noise,size=full_set.shape))/255
        g_truth=np.concatenate((np.ones(n_images*3),np.zeros(n_images*3)))
        g_truth=to_categorical(g_truth, num_classes=2)
        full_set=full_set.reshape(-1,self.imgsize*224,self.imgsize*224,1)
        return full_set, g_truth
    
    def testsetmaker(self,set_size,n_images,noise):
        full_set=np.concatenate((self.multistimcreatorpres(set_size,n_images),self.multistimcreatorabs(set_size,n_images)))
        gc.collect()
        full_set=(full_set+np.random.normal(0,noise,size=full_set.shape))/255
        g_truth=np.concatenate((np.ones(n_images),np.zeros(n_images)))
        g_truth=to_categorical(g_truth, num_classes=2)
        full_set=full_set.reshape(-1,self.imgsize*224,self.imgsize*224,1)
        return full_set, g_truth 


    def items(self):
        center=self.get_centre(self.N)
        d=self.imgsize*(self.length+2)
        image_to_draw_on=128*np.ones((self.imgsize*224,self.imgsize*224))
        image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,self.N,self.dist_luminance)
        luminance_dist_item=image_to_draw_on[int(center[1]-d):int(center[1]+d),int(center[0]-d):int(center[0]+d)]/255
        image_to_draw_on=128*np.ones((self.imgsize*224,self.imgsize*224))
        image_to_draw_on=self.draw_slope(image_to_draw_on,self.dist_angle,self.N,self.target_luminance)
        angle_dist_item=image_to_draw_on[int(center[1]-d):int(center[1]+d),int(center[0]-d):int(center[0]+d)]/255
        image_to_draw_on=128*np.ones((self.imgsize*224,self.imgsize*224))
        image_to_draw_on=self.draw_slope(image_to_draw_on,self.target_angle,self.N,self.target_luminance)
        target_dist_item=image_to_draw_on[int(center[1]-d):int(center[1]+d),int(center[0]-d):int(center[0]+d)]/255
        return target_dist_item,luminance_dist_item,angle_dist_item



    def makedicts(self):
        setsizes = [*range(1,self.N+1)]
        singlesettypedictluminance={}
        singlesettypedictangle={}
        singlesettypedictconj={}


        for i in setsizes:
            temppres=[1]
            tempabs=[]
            while len(temppres)<i:
                temppres.append(2)
            while len(tempabs)<i:
                tempabs.append(2)
            singlesettypedictluminance['pset'+str(i)]=temppres
            singlesettypedictluminance['aset'+str(i)]=tempabs

        for i in setsizes:
            temppres=[1]
            tempabs=[]
            while len(temppres)<i:
                temppres.append(3)
            while len(tempabs)<i:
                tempabs.append(3)
            singlesettypedictangle['pset'+str(i)]=temppres
            singlesettypedictangle['aset'+str(i)]=tempabs
        
        for i in setsizes:

            temppres=[1]
            tempabs=[]
            for j in range(i):
                temppres.append(2)
                tempabs.append(2)
            for j in range(i):
                temppres.append(3)
                tempabs.append(3)
            singlesettypedictconj['pset'+str(i)]=temppres
            singlesettypedictconj['aset'+str(i)]=tempabs

        return singlesettypedictconj ,singlesettypedictluminance, singlesettypedictangle

    
    def makeallcombs(self):
        try:
            settypedictconj=np.load('Data/itemCombs/'+str(self.N)+'settypedictconj.npy',allow_pickle=True)[()]
            settypedictluminance=np.load('Data/itemCombs/'+str(self.N)+'settypedictluminance.npy',allow_pickle=True)[()]
            settypedictangle=np.load('Data/itemCombs/'+str(self.N)+'settypedictangle.npy',allow_pickle=True)[()]
        except:
            singlesettypedictconj ,singlesettypedictluminance, singlesettypedictangle = self.makedicts()
            settypedictconj={}
            settypedictluminance={}
            settypedictangle={}
            
            for i in singlesettypedictluminance.keys():
                settypedictluminance[i]= np.unique(np.array(list(idp(singlesettypedictluminance[i]))),axis=0)
            for i in singlesettypedictangle.keys():
                settypedictangle[i]= np.unique(np.array(list(idp(singlesettypedictangle[i]))),axis=0)
            
            for i in singlesettypedictconj.keys():
                if str(i[0])=='a':
                    settypedictconj[i]= np.unique(np.unique(np.array(list(idp(singlesettypedictconj[i]))),axis=0)[:,:int(i[4:])],axis=0)
                elif str(i[0])=='p':
                    temp= np.unique(np.unique(np.array(list(idp(singlesettypedictconj[i]))),axis=0)[:,:int(i[4:])],axis=0)
                    temp2=[]
                    for j in temp:
                        if 1 in j:
                            temp2.append(j)
                    settypedictconj[i]=np.array(temp2)

        return settypedictconj, settypedictluminance,settypedictangle