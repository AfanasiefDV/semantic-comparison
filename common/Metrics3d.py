import plotly.express as px
import pandas as pd

def img3d(x, y, z, markers):
    colors = []
    for i in range(len(markers)):
        colors.append('Blues' if markers[i] else 'Reds')
    d = {'keywords': x, 'text': y, 'sim': z, 'colors': colors}
    df = pd.DataFrame(d)
    fig = px.scatter_3d(df, x='keywords', y='text', z='sim',
                        color='colors')
    fig.show()
