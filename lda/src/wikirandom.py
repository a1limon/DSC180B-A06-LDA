# wikirandom.py: Functions for downloading random articles from Wikipedia
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import urllib.request, urllib.error, urllib.parse, re
from bs4 import BeautifulSoup
import time
import threading
import pickle
import bz2


def get_random_wikipedia_article():
    """
    Downloads a randomly selected Wikipedia article (via
    http://en.wikipedia.org/wiki/Special:Random) and strips out (most
    of) the formatting, links, etc. 
    This function is a bit simpler and less robust than the code that
    was used for the experiments in "Online VB for LDA."
    """
    failed = True
    while failed:
        articletitle = None
        failed = False
        try:
            req = urllib.request.Request('http://en.wikipedia.org/wiki/Special:Random',
                                  None, { 'User-Agent' : 'x'})
            f = urllib.request.urlopen(req)
            while not articletitle:
                line = f.readline().decode('utf-8')  # convert bytes to string
                result = re.search(r'title="Edit this page" href="/w/index.php\?title=(.*)&amp;action=edit"/\>', line)
                if (result):
                    articletitle = result.group(1)
                    break
                elif (len(line) < 1):
                    sys.exit(1)

            all = f.read()
        except (urllib.error.HTTPError, urllib.error.URLError):
            print('oops. there was a failure downloading %s. retrying...' \
                % articletitle)
            failed = True
            continue
        print('downloaded %s. parsing...' % articletitle)

        try:
            soup = BeautifulSoup(all,'lxml')  # use lxml parser for best performance
            # extract text from paragraph elements
            paras = []
            for paragraph in soup.find_all('p'):
                paras.append(str(paragraph.text))
     
            all = [p for p in paras]
            all = ' '.join(all)

            # remove footnotes, superscripts 
            all = re.sub(r"\[.*?\]+", '', all)
            all = all.replace('\n', '')
      
        except:
            # Something went wrong, try again. (This is bad coding practice.)
            print('oops. there was a failure parsing %s. retrying...' \
                % articletitle)
            failed = True
            continue

    return(all, articletitle)

class WikiThread(threading.Thread):
    articles = list()
    articlenames = list()
    lock = threading.Lock()

    def run(self):
        (article, articlename) = get_random_wikipedia_article()
        WikiThread.lock.acquire()
        WikiThread.articles.append(article)
        WikiThread.articlenames.append(articlename)
        WikiThread.lock.release()

def get_random_wikipedia_articles(n):
    """
    Downloads n articles in parallel from Wikipedia and returns lists
    of their names and contents. Much faster than calling
    get_random_wikipedia_article() serially.
    """
    maxthreads = 8
    WikiThread.articles = list()
    WikiThread.articlenames = list()
    wtlist = list()
    for i in range(0, n, maxthreads):
        print('downloaded %d/%d articles...' % (i, n))
        for j in range(i, min(i+maxthreads, n)):
            wtlist.append(WikiThread())
            wtlist[len(wtlist)-1].start()
        for j in range(i, min(i+maxthreads, n)):
            wtlist[j].join()
    return (WikiThread.articles, WikiThread.articlenames)

if __name__ == '__main__':
    # demo
    
    t0 = time.time()
    n = 1  # number of articles to download
    (articles, articlenames) = get_random_wikipedia_articles(n)

    with bz2.BZ2File('wikiarticles.pbz2', 'w') as f:
        pickle.dump(articles, f)

    with bz2.BZ2File('wikiarticlenames.pbz2', 'w') as f:
        pickle.dump(articlenames, f)
    
    t1 = time.time()
    print('took %f' % (t1 - t0))