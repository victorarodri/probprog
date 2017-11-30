# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud

plt.style.use('ggplot')


def gen_wordcloud(tokens, probs):

    def red_color_func(word, font_size, position, orientation,
                       random_state=None, **kwargs):
        return "hsl(0, 100%%, %d%%)" % random.randint(25, 55)

    def green_color_func(word, font_size, position, orientation,
                         random_state=None, **kwargs):
        return "hsl(90, 100%%, %d%%)" % random.randint(27, 37)

    def blue_color_func(word, font_size, position, orientation,
                        random_state=None, **kwargs):
        return "hsl(200, 100%%, %d%%)" % random.randint(30, 60)

    data_types = ['note', 'lab', 'med']

    colors = {0: red_color_func,
              1: green_color_func,
              2: blue_color_func}

    f, axn = plt.subplots(1, 3)

    for j, ax in enumerate(axn.flatten()):

        tokens_probs = dict((t, p) for t, p in zip(tokens[j], probs[j]))

        wordcloud = WordCloud(background_color="white", relative_scaling=0.5)
        wordcloud.generate_from_frequencies(tokens_probs)
        wordcloud.recolor(color_func=colors[j], random_state=3)

        ax.axis('off')
        ax.imshow(wordcloud)
        ax.set_title(data_types[j], fontsize=50)

    f = plt.gcf()
    f.set_figwidth(36.0)
    f.set_figheight(36.0)

    plt.tight_layout()
    plt.show()
    plt.close('all')


def gen_colormeshes(probs, colormaps):

    for s in range(len(colormaps)):
        plt.pcolormesh(np.log(probs[s].params[-1].eval()), cmap=colormaps[s])
        plt.tick_params(labelsize='large')
        plt.xlabel('Token ID')
        plt.ylabel('Phenotype ID')
        plt.colorbar()

        f = plt.gcf()
        f.set_figwidth(14.0)
        f.set_figheight(5.5)

        plt.show()


def main():
    """Empty main function."""
    return True


if __name__ == '__main__':
    main()
