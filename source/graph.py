#!/usr/bin/env python
# coding: utf-8


import os
try:
    os.environ.pop('http_proxy')
    os.environ.pop('https_proxy')
except KeyError:
    pass

from jupyter_dash.comms import _send_jupyter_config_comm_request
_send_jupyter_config_comm_request()



import sys
sys.path.append("../source")

from jupyter_dash import JupyterDash

import df_db as df_db
#JupyterDash.infer_jupyter_proxy_config()

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import dash_table
import math
import yaml
from numpy import pi, sin, cos
import networkx as nx


#####################################################


with open(r'../config/secrets.yaml') as file:
    secrets = yaml.load(file, Loader=yaml.FullLoader)


mapbox_access_token = secrets['mapbox']




#Comm_list needed for community selection
db_conn = df_db.db_df_loader('micro_mobility')

Community_Query = """
SELECT community
FROM communities
"""

Comm_list = sorted(list(db_conn.query_df(Community_Query).community)) #x.lower() for x in 


# defines layout including mapbox background, centering and appropriate zoom
def layout_mapper(points):
    lat_center = points.latitude.mean()
    long_center = points.longitude.mean()
    layout_map = dict(
        autosize=True,
        height=500,
        width=1000,
        font=dict(color="#191A1A"),
        titlefont=dict(color="#191A1A", size=14),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        hovermode="closest",
        plot_bgcolor='#fffcfc',
        paper_bgcolor='#fffcfc',
        legend=dict(font=dict(size=10), orientation='h'),
        mapbox=dict(
            accesstoken=mapbox_access_token,
            style="outdoors",
            center=dict(
                lon = long_center,
                lat = lat_center
            ),
            zoom=10,
        )
    )
    return layout_map



#utilizes networkx graph with geographically tied layout to create plotly/ mapbox plot
def networkGraph(G, layout_map, trips_gpd, pos):

    

    # edges trace: assigning weights for line thickness and creating curvature
    edge_x = []
    edge_y = []
    coords = []
    coords_x = []
    coords_y = []
    traces=[]
    weights = list(trips_gpd.Trip_count)
    weights_adj = [max(j/max(weights)*10,.25) for j in weights]
    
    for i, edge in enumerate(G.edges()):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        #mapbox does not have a native solution to create curved lines
        #next step is to assign midpoint based on other points to minimize crossing lines with points
        try:
            R = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)/2
            m = 1/(y1-y0)/(x1-x0)
            ang = math.atan(m)
            center_lon = np.mean([x0,x1])
            center_lat = np.mean([y0,y1])
            yb = R*sin(ang) + center_lat
            xb = R*cos(ang) + center_lon

            def recta(x1, y1, x2, y2):
                a = (y1 - y2) / (x1 - x2)
                b = y1 - a * x1
                return (a, b)

            def curva_b(xa, ya, xb, yb, xc, yc):
                (x1, y1, x2, y2) = (xa, ya, xb, yb)
                (a1, b1) = recta(xa, ya, xb, yb)
                (a2, b2) = recta(xb, yb, xc, yc)
                puntos = []

                for i in range(0, 1000):
                    if x1 == x2:
                        continue
                    else:
                        (a, b) = recta(x1, y1, x2, y2)
                    x = i*(x2 - x1)/1000 + x1
                    y = a*x + b
                    puntos.append((x,y))
                    x1 += (xb - xa)/1000
                    y1 = a1*x1 + b1
                    x2 += (xc - xb)/1000
                    y2 = a2*x2 + b2
                return puntos

            puntos = curva_b(x0,y0,xb,yb,x1,y1)

            #circle_lon, circle_lat = hanging_line([x0,y0],[x1,y1])

            for lo, la in puntos: #zip(list(circle_lon), list(circle_lat)):
                coords.append([lo, la]) 
                coords_x.append(lo)
                coords_y.append(la)
            coords.append(None)
            #coords_x.append(None)
            #coords_y.append(None)
        except:
            continue
        
    
        traces.append(go.Scattermapbox(lon=coords_x,
                                       lat=coords_y,
                                       line=dict(color='red',
                                                 width=weights_adj[i]
                                                ),
                                       hoverinfo='none',
                                       showlegend=False,
                                       mode='lines')
                     )
                      
        edge_x = []
        edge_y = []
        coords = []
        coords_x = []
        coords_y = []
    

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scattermapbox(
        lon = node_x,
        lat = node_y,
        text=text,
        mode = 'markers+text',
        showlegend=False,
        opacity = 1
    )
        
    traces.append(node_trace)

    # figure
    fig = go.Figure(data=traces, layout=layout_map)
    
    
    return fig




# Code populating graph
def query_pop(C_orig_list, C_end_list):
    orig_list = C_orig_list.copy()
    orig_list.extend(['placeholder1','placeholder2'])
    orig_l = tuple(x.lower() for x in orig_list)
    
    dest_list = C_end_list.copy()
    dest_list.extend(['placeholder1','placeholder2'])
    dest_l = tuple(x.lower() for x in dest_list)
    
    IND_Query = """
    SELECT *
    FROM trips
    WHERE  LOWER("Start_Community_Area_Name")
    IN {}
    AND LOWER("End_Community_Area_Name")
    IN {}
    """.format(orig_l, dest_l)


    trips = db_conn.query_df(IND_Query)
    trips = trips.dropna(subset = ['Start_Centroid_Latitude', 'End_Centroid_Latitude'])
    print(trips.head())
    point_orig = trips[
        ['Start_Community_Area_Name',
         'Start_Centroid_Latitude',
         'Start_Centroid_Longitude']
    ].rename(
        columns = {'Start_Community_Area_Name':'community',
                   'Start_Centroid_Latitude':'latitude',
                   'Start_Centroid_Longitude':'longitude'}
    )

    point_dest = trips[
        ['End_Community_Area_Name',
         'End_Centroid_Latitude',
         'End_Centroid_Longitude']
    ].rename(
        columns = {'End_Community_Area Name':'community',
                   'End_Centroid_Latitude':'latitude',
                   'End_Centroid_Longitude':'longitude'}
    )

    points = pd.concat([point_orig, point_dest])

    trips_loc = trips.dropna(subset = ['Start_Community_Area_Name','End_Community_Area_Name'])

    trips_gpd = trips_loc[
        ['Start_Community_Area_Name',
         'End_Community_Area_Name',
         'Trip_ID']
    ].groupby(
        ['Start_Community_Area_Name',
         'End_Community_Area_Name']
    ).count().rename(
        columns = {'Trip_ID': 'Trip_count'}
    ).sort_values(by = 'Trip_count', ascending = False).reset_index()
    
    params = [(x, y, {'weight': v}) for x,y,v in zip(trips_gpd['Start_Community_Area_Name'], 
                                                     trips_gpd['End_Community_Area_Name'], 
                                                     trips_gpd['Trip_count'])]
    
    g = nx.DiGraph(params)
    
    orig = trips_loc[['Start_Community_Area_Name',
                      'Start_Centroid_Longitude',
                      'Start_Centroid_Latitude']
                    ].rename(
        columns = {'Start_Community_Area_Name':'node','Start_Centroid_Longitude':'x','Start_Centroid_Latitude':'y'}
    )

    dest = trips_loc[['End_Community_Area_Name',
                      'End_Centroid_Longitude',
                      'End_Centroid_Latitude']
                    ].rename(
        columns = {'End_Community_Area_Name':'node','End_Centroid_Longitude':'x','End_Centroid_Latitude':'y'}
    )

    nodes = pd.concat([orig, dest])

    nodes_mean = nodes.groupby('node').mean().reset_index()
    
    pos = dict((n,(x,y)) for n,x,y in zip(nodes_mean.node,nodes_mean.x,nodes_mean.y))

    layout_map = layout_mapper(points)
    
    return networkGraph(g, layout_map, trips_gpd, pos)


#Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/amyoshino/pen/jzXypZ.css'])#
#app.title = "Chicago Scooter Network"

app.layout = html.Div(children = [
    html.H1("Chicago Scooter Trips: 2019"),
    html.H3("Origin Communities"),
    dcc.Dropdown(id='Origin',
                 options=[{'label':x, 'value':x} for x in Comm_list],
                 value = Comm_list[0:6],
                 multi=True,
                 ),
    html.H3("Destination Communities"),
    dcc.Dropdown(id='Destination',
                 options=[{'label':x, 'value':x} for x in Comm_list],
                 value = Comm_list,
                 multi=True,
                ),
    html.H3("Trip Density"),
    dcc.Graph(id='my-graph'),
    ]
)



@app.callback(
    Output('my-graph', 'figure'),
    [Input('Origin', 'value'),
     Input('Destination', 'value')],
)
def update_output(value1, value2):
    return query_pop(value1, value2)

if __name__ == "__main__":
    app.run_server(
        # host="your-local-ip-here"
        #host="192.168.0.17",
        #port="8088",
        use_reloader=True,
        debug=True,
    )







