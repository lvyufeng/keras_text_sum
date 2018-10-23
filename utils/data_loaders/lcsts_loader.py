from bs4 import BeautifulSoup
import jieba

def load_text(path):
    f = open(path)
    soup = BeautifulSoup(f,'lxml')
    f.close()

    summary = soup.select('doc > summary')
    short_text = soup.select('doc > short_text')

    summary = [i.text.strip('\n').strip() for i in summary]
    short_text = [i.text.strip('\n').strip() for i in short_text]

    return summary,short_text

def load_split_word(path):
    f = open(path)
    soup = BeautifulSoup(f, 'lxml')
    f.close()

    summary = soup.select('doc > summary')
    short_text = soup.select('doc > short_text')

    summary = [' '.join(jieba.cut(i.text.strip('\n').strip())) for i in summary]
    short_text = [' '.join(jieba.cut(i.text.strip('\n').strip())) for i in short_text]

    return summary, short_text



# load_split_word('/home/lv/data_set/LCSTS2.0/DATA/PART_III.txt')