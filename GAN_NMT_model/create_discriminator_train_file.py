import random

sf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/test.en', 'r')
machine_tf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/out_test.fr', 'r')
human_tf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/test.fr', 'r')

train_en = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_discriminator/test.en', 'w')
train_fr = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_discriminator/test.fr', 'w')
train_label = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_discriminator/test.label', 'w')

#for idx, (sline, mline, hline) in enumerate(zip(sf, machine_tf, human_tf)):
#    sline = sline.strip()
#    mline = mline.strip()
#    hline = hline.strip()
#
#    # Exception if empty line found
#    if sline == "" or mline == "" or hline == "":
#        continue
#
#    train_en.write(sline + '\n')
#    train_en.write(sline + '\n')
#    
#    train_fr.write(hline + '\n')
#    train_fr.write(mline + '\n')
#    
#    train_label.write('1 0' + '\n')
#    train_label.write('0 1' + '\n')
    
    
    
for idx, (sline, mline, hline) in enumerate(zip(sf, machine_tf, human_tf)):
    sline = sline.strip()
    mline = mline.strip()
    hline = hline.strip()

    # Exception if empty line found
    if sline == "" or mline == "" or hline == "":
        continue    
    
    word = ''
    hseq = []
    for w in hline.split(' '):
        word = word + " " + w
        hseq.append(word)
        
    word = ''
    mseq = []
    for w in mline.split(' '):
        word = word + " " + w
        mseq.append(word)
    
    train_en.write(sline + '\n')
    train_en.write(sline + '\n')
    
    train_fr.write(random.choice(hseq) + '\n')
    train_fr.write(random.choice(mseq) + '\n')
    
    train_label.write('1 0' + '\n')
    train_label.write('0 1' + '\n')