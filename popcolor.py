import glob
import numpy as np
import matplotlib.pyplot as plt 
from cv2 import kmeans, KMEANS_RANDOM_CENTERS, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER
from PIL import Image


class vegnn:
    k = 6
    data = {}
    def __init__(self, path, save_folder, pad=True):
        images = glob.iglob(path)
        for image in images:
            self.imgb = Image.open(image)
            self.pad = pad
            self.img_posterized = self.imgb.quantize((self.k//2)*self.k**2, 1)
            self.img_posterized = self.img_posterized.convert("RGB", palette=Image.ADAPTIVE, colors=self.k**3)
            self.img_posterized = np.array(self.img_posterized)

            ret, self.label, self.center = self._clustering()
            self.data[image[image.index('/')+1:]] = self.center

            if (self.imgb.width-self.imgb.height) > (self.imgb.width+self.imgb.height)/10: 
                self.figdraw(flag=0, w=50, h=15, r=1, c=2, an1='C', an2='W', bb=(1,0,-0.5,1), loc="center left")
            else: 
                self.figdraw(flag=1, w=15, h=50, r=2, c=1, an1='S', an2='N', bb=(0.5,1.2), loc='lower center')

            self.conc_pic(name=image, save_folder=save_folder)

    def __getitem__(self, key):
        return self.data[key]

    def _clustering(self):
        return kmeans(
                    self.img_posterized.reshape(-1, 3).astype('float32'), 
                    self.k, 
                    None, 
                    (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 10, 1), 
                    0,
                    KMEANS_RANDOM_CENTERS
                        )

    def _hist(self):
        (hist, _) = np.histogram(self.label, bins=self.k)
        mask = np.argsort(hist)
        self.center = self.center.reshape(-1, 3)[mask]
        return hist[mask]


    def figdraw(self, flag, w, h, r, c, an1, an2, bb, loc):
        self.flag = flag
        fig = plt.figure()
        fig.set_size_inches(w, h)

        ax1 = plt.subplot(r, c, 2, aspect="equal", anchor=an1)
        ax2 = plt.subplot(r, c, 1, aspect="equal", anchor=an2)

        h = self._hist()
        self.center = self.center.astype('uint8')
        colors = np.array(['#{:02X}{:02X}{:02X}'.format(x[0],x[1],x[2]) for x in self.center])
        wedges, _ = ax1.pie(h, colors=colors, startangle=90, radius=1.25)
        ax1.legend(wedges, colors, loc=loc, bbox_to_anchor=bb, fontsize=90+self.flag*20, 
                    labelspacing=0.75+self.flag*((self.k-h.size)/(self.k*2)))
    
        ax2.imshow(self.img_posterized)
        ax2.axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(w, h, 3)
        plt.close()

        self.figr = Image.frombytes("RGB", (w, h), buf.tostring())
        
    def conc_pic(self, name, save_folder):
        temp = np.array(self.imgb)

        if self.pad:
            pv = max(self.imgb.width, self.imgb.height)//85
            temp = np.pad(temp, ((pv,pv), (pv,pv), (0,0)), 'constant', constant_values=255)

        u = Image.fromarray(temp)
        d = self.figr

        w1, h1 = u.width, u.height
        w2, h2 = d.width, d.height

        if self.flag:
            if h1<h2: d = d.resize((int(w2*(h1/h2)), h1), Image.LANCZOS)
            if h1>h2: u = u.resize((int(w1*(h2/h1)), h2), Image.LANCZOS)
            self.ans = np.hstack((np.asarray(u),np.asarray(d)))
        else:    
            if w1<w2: d = d.resize((w1, int(h2*(w1/w2))), Image.LANCZOS)
            if w1>w2: u = u.resize((w2, int(h1*(w2/w1))), Image.LANCZOS)
            self.ans = np.vstack((np.asarray(u),np.asarray(d)))
        
        self.ans = Image.fromarray(self.ans)
        self.ans.save(save_folder+'/'+name[name.index('/')+1:])
