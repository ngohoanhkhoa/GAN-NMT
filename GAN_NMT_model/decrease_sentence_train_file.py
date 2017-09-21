sf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data/train.en', 'r')
tf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data/train.fr', 'r')

train_en = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data/train_research.en', 'w')
train_fr = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data/train_research.fr', 'w')

sentence_num = 0
for (sline, tline) in zip(sf, tf):
    
    if sentence_num == 50:
        break
    
    sline = sline.strip()
    tline = tline.strip()

    # Exception if empty line found
    if sline == "" or tline == "":
        continue

    train_en.write(sline + '\n')
    train_fr.write(tline + '\n')
    sentence_num+= 1
    
sf.close
tf.close
train_en.close
train_fr.close