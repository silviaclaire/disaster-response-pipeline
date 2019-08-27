import pandas as pd
from plotly.graph_objs import Bar


def return_graphs(df):
    """Extract data needed for visualization.

    Args:
        df(pd.DataFrame): data to be visualized

    Returns:
        graphs(list): a list of visualization dicts
    """
    # message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # message categories
    categories = df.loc[:, 'related':]
    category_names = list(categories.columns)
    category_counts = categories.sum(axis=0)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
    ]

    return graphs
