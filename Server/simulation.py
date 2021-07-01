#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass


import carla
from agents.navigation.controller import VehiclePIDController
import math
import random
import pandas as pd
from math import sqrt
import time
from PixelManner import PixelMapper
from matplotlib import colors
from random import randrange

import numpy as np
from sympy import *
x, y, w, h = symbols('x y w h')

height = 1080
width = 1920
fps =(1000/60)
angle = 120
altura=0.1

pixels_camera_wh =[[251,309],[352,249],[1108,381],[898,310],[1039,307]]
points_carla_xy =[[-82.16,12.17],[-84.92,9.95],[-66.53,-15.70],[-71.5,-14.6],[-83.3,6.2]]
#x=-30.895,y=40.698, z=18
point_camera=carla.Transform(carla.Location(x=-49.360,y=-5.110, z=9), carla.Rotation(yaw=120, pitch=0))

eqs=[]
for i in range(0,len(pixels_camera_wh)):
    eqs.append(points_carla_xy[i][0]*x+points_carla_xy[i][1]*y-pixels_camera_wh[i][0]*w-pixels_camera_wh[i][1]*h)

eqh=solve(eqs,exclude=[h])
eqw=solve(eqs,exclude=[w])
import numpy as np


points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1]]
points_carla = [[-82,13.1,1],[-86.2,10.3,1],[-83.3,5.7,1],[-71.9,-14,1],[-69.7,-18.8,1],[-65.7,-16.1,1]]

points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1]]
points_carla = [[-65.6,-16,1],[-69.5,-18.6,1],[-72,-14.8,1],[-83.3,6,1],[-85.8,9.5,1],[-81.7,13,1]]

points_pixel = [[533,313,1],[921,320,1],[1589,377,1],[1656,438,1]]
points_carla = [[-65.6,-16,1],[-69.5,-18.6,1],[-85.8,9.5,1],[-81.7,13,1]]

points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1]]
points_carla = [[-82,13.1,1],[-86.2,10.3,1],[-83.3,5.7,1],[-71.9,-14,1],[-69.7,-18.8,1],[-65.7,-16.1,1]]


points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1],[651,246,1],[1279,227,1]]
points_carla = [[-61.8,-16.5,1],[-72.3,-22.3,1],[-77.1,-12.3,1],[-84.7,1.8,1],[-90.5,12.5,1],[-81.9,19.2,1],[-82.8,-31.4,1],[-103.8,0.1,1]]

points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1],[651,246,1],[1279,227,1],[720,895,1],[1258,920,1]]
points_carla = [[-80,19.5,1],[-90.5,10.4,1],[-85.7,1.3,1],[-77.1,-14.5,1],[-71.4,-25.3,1],[-61.8,-17,1],[-100,1.3,1],[-85.7,-36.9,1],[-66.6,14.5,1],[-58,-1.2,1]]


points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1],[651,246,1],[1279,227,1],[720,895,1],[1258,920,1]]
points_carla = [[-80,19.5,1],[-90.5,10.4,1],[-85.7,1.3,1],[-77.1,-14.5,1],[-71.4,-25.3,1],[-61.8,-17,1],[-100,1.3,1],[-85.7,-36.9,1],[-66.6,14.5,1],[-58,-1.2,1]]

points_pixel = [[533,313,1],[921,320,1],[1032,309,1],[1569,270,1],[1589,377,1],[1656,438,1],[651,246,1],[1279,227,1],[720,895,1],[1258,920,1],[858,395,1],[1084,405,1]]
points_pixel = [[533,312,1],[900,310,1],[1488,291,1],[1464,332,1],[1457,326,1],[1652,440,1],[720,228,1],[1279,230,1],[723,895,1],[1235,910,1],[1057,407,1],[1086,403,1],[966,317,1],
                [750,261,1],[1147,228,1],[88,852,1],[395,898,1],[1565,934,1],[1818,919,1],[617,395,1],[1366,412,1]]
points_carla = [[-80,19.5,1.3],[-90.5,10.4,1.3],[-85.7,1.3,1.3],[-77.1,-14.5,1.3],[-71.4,-25.3,1.3],[-61.8,-17,1.3],[-105.8,-2.1,1.3],[-81.9,-34.4,1.3],[-66.6,14.5,1.3],[-58,-1.2,1.3],[-76.1,7.1,1.3],[-66.6,-7,1.3],[-79,-7.9,1.3],
                [-87.6,6.2,1.3],[-74.2,-19.5,1.3],[-71.4,21.2,1.3],[-69.4,17.8,1.3],[-55.1,-2.1,1.3],[-55.1,-5.4,1.3],[-80,13.7,1.3],[-64.7,-11.2,1.3]]

points_pixel = [[533,312,1],[900,310,1],[1488,291,1],[1464,332,1],[1457,326,1],[1652,440,1],[723,895,1],[1235,910,1],[1057,407,1],[1086,403,1],
                [717,311,1],[1224,310,1],[88,852,1],[395,898,1],[1565,934,1],[1818,919,1],[614,397,1],[1367,411,1]]
points_carla = [[-47.6,30,1.3],[-53.4,39.7,1.3],[-63.8,30.8,1.3],[-72.4,25.1,1.3],[-82.9,16.2,1.3],[-78.1,5.7,1.3],[-49.5,5.7,1.3],[-59.1,-0.8,1.3],[-59.1,21.1,1.3],[-67.6,13.8,1.3],
                [-59.1,34.9,1.3],[-77.2,19.5,1.3],[-42.9,11.3,1.3],[-45.7,8.9,1.3],[-62.9,-3.3,1.3],[-66.7,-5.7,1.3],[-54.3,24.3,1.3],[-73.4,9.7,1.3]]


points_pixel = [[533,312,1],[900,310,1],[1488,291,1],[1464,332,1],[1457,326,1],[1652,440,1],[720,228,1],[1279,230,1],[723,895,1],[1235,910,1],[1057,407,1],[1086,403,1],
                [750,261,1],[1147,228,1],[88,852,1],[395,898,1],[1565,934,1],[1818,919,1]]
points_carla = [[-47.100,30.230,0.1],[-51.350,39.440,0.1],[-63.480,30.760,0.1],[-72.720,25.150,0.1],[-83.690,15.180,0.1],[-80.090,5.890,0.1],[-60.186,55.056,0.1],[-90.056,25.803,0.1],[-48.873,6.368,0.1],[-59.223,-0.561,0.1],[-57.603,21.487,0.1],[-67.682,14.378,0.1],
                [-58.1,34.9,0.1],[-80,20.3,0.1],[-42.9,11.3,0.1],[-45.7,8.9,0.1],[-62.9,-3.3,0.1],[-66.7,-5.7,0.1]]

points_pixel = [
                [870,352,1],[933,460,1],
                [533,312,1],[900,310,1],[1488,291,1],[1464,332,1],[1457,326,1],[1652,440,1],[723,895,1],[1235,910,1],[1057,407,1],[1086,403,1],
                [663,310,1],[1011,238,1],[1171,312,1],[1286,313,1],[88,852,1],[395,898,1],[1565,934,1],[1818,919,1],
                [605,405,1],[1134,379,1],[1455,411,1],[1460,407,1]]
points_carla = [
                [-60.989,25.977,0.1],[-70.075,19.143,0.1],
                [-47.100,30.230,0.1],[-51.350,39.440,0.1],[-63.480,30.760,0.1],[-72.720,25.150,0.1],[-83.690,15.180,0.1],[-80.090,5.890,0.1],[-48.873,6.368,0.1],[-59.223,-0.561,0.1],[-57.603,21.487,0.1],[-67.682,14.378,0.1],
                [-55.277,36.715,0.1],[-60.214,32.985,0.1],[-74.695,21.904,0.1],[-79.742,17.845,0.1],[-41.674,11.318,0.1],[-46.354,8.168,0.1],[-61.922,-2.361,0.1],[-66.872,-5.691,0.1],
                [-50.120,26.841,0.1],[-54.620,23.330,0.1],[-70.654,11.875,0.1],[-74.914,8.481,0.1]]


points_pixel = [
                [880,314,1],[1055,321,1],[778,913,1],[1251,907,1]]

points_carla = [
                [-62.503,31.869,0.1],[-72.869,24.732,0.1],[-48.873,6.368,0.1],[-59.223,-0.561,0.1]]



points_pixel = [
                [387,398,1],[538,316,1],[886,318,1],[1051,318,1],[1455,323,1],[1641,430,1],
                [673,309,1],[785,308,1],[1161,302,1],[1256,304,1],
                [708,894,1],[1256,928,1],
                [67,861,1],[423,887,1],[1612,921,1],[1862,944,1]
                ]
points_carla = [
                [-48.230,33.780,0.1],[-51.970,39.220,0.1],[-63.340,30.510,0.1],[-70.290,25.270,0.1],[-83.690,15.180,0.1],[-80.160,7.710,0.1],
                [-54.710,37.180,0.1],[-60.570,32.560,0.1],[-72.790,23.150,0.1],[-79.680,17.980,0.1],
                [-50.500,5.410,0.1],[-56.294,1.013,0.1],
                [-41.190,12.290,0.1],[-46.900,8.030,0.1],[-60.040,-1.770,0.1],[-66.900,-6.620,0.1]
                
                ]


num_prom=5

class SpecialTransform:
    transform=None
    def __init__(self, transform):
        self.transform = transform

def normalizedpixel(u,v):
    print(u)
    return [u-(width/2),v-(height/2)]

def normalizedPointsPixel(m):
    final=[]
    for i in m:
        t= normalizedpixel(i[0],i[1])
        final.append([t[0],t[1],1])
    return final
def getMatrix2():
    pc=[]
    r=[]
    for j in points_pixel:
        pc.append([j[0]**2, j[0]*j[1],j[1]**2, j[0],j[1],1 ])
    for i in points_carla:
        r.append([i[0],i[1]])
    pc = np.array(pc)
    r=np.array(r)
    

    A=np.linalg.pinv(np.transpose(pc).dot(pc)).dot(np.transpose(pc)).dot(r)
    return A
def getMatrix():

    rxx= sum(points_pixel[i][0]*points_carla[i][0] for i in range(0,len(points_pixel)))
    rxy=sum(points_pixel[i][1]*points_carla[i][0] for i in range(0,len(points_pixel)))
    rxz=sum(points_pixel[i][2]*points_carla[i][0] for i in range(0,len(points_pixel)))
    tx=sum(x[0] for x in points_carla)

    r_x= np.array([rxx,rxy,rxz,tx])

    ryx= sum(points_pixel[i][0]*points_carla[i][1] for i in range(0,len(points_pixel)))
    ryy=sum(points_pixel[i][1]*points_carla[i][1] for i in range(0,len(points_pixel)))
    ryz=sum(points_pixel[i][2]*points_carla[i][1] for i in range(0,len(points_pixel)))
    ty=sum(x[1] for x in points_carla)

    r_y= np.array([ryx,ryy,ryz,ty])

    rzx= sum(points_pixel[i][0]*points_carla[i][2] for i in range(0,len(points_pixel)))
    rzy=sum(points_pixel[i][1]*points_carla[i][2] for i in range(0,len(points_pixel)))
    rzz=sum(points_pixel[i][2]*points_carla[i][2] for i in range(0,len(points_pixel)))
    tz=sum(x[2] for x in points_carla)

    r_z= np.array([rzx,rzy,rzz,tz])


    A = np.array([
    [sum( x[0]**2 for x in points_pixel), sum(x[0]*x[1] for x in points_pixel),sum(x[0]*x[2] for x in points_pixel),sum(x[0] for x in points_pixel)],
    [sum(x[0]*x[1] for x in points_pixel),sum( x[1]**2 for x in points_pixel),sum(x[1]*x[2] for x in points_pixel),sum(x[1] for x in points_pixel)],
    [sum(x[0]*x[2] for x in points_pixel),sum(x[1]*x[2] for x in points_pixel),sum( x[2]**2 for x in points_pixel),sum(x[2] for x in points_pixel)],
    [sum(x[0] for x in points_pixel),sum(x[1] for x in points_pixel),sum(x[2] for x in points_pixel),len(points_pixel)]
    ])


    TAB = np.array([
    np.linalg.pinv(A).dot(r_x),
    np.linalg.pinv(A).dot(r_y),
    np.linalg.pinv(A).dot(r_z),
    [0,0,0,1]
    ])
    return TAB

def convertPoint(mat,x,y):
    cp=mat.dot(np.array([x,y,1,1]))
    return [cp[0],cp[1]]

def convertPoint2(mat,x,y):
    xf=mat[0][0]*(x**2)+ mat[1][0]*(x*y)+ mat[2][0]*(y**2)+ mat[3][0]*(x)+ mat[4][0]*(y)+ mat[5][0]
    yf=mat[0][1]*(x**2)+ mat[1][1]*(x*y)+ mat[2][1]*(y**2)+ mat[3][1]*(x)+ mat[4][1]*(y)+ mat[5][1]
    return [xf,yf]

points_pixel=normalizedPointsPixel(points_pixel)
print("Configure")
mt=getMatrix()
print(mt)


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))





def get_coord_normals2(wc,hc, mape):
    y_solver=float(solve([eqh[y]-y,h-hc])[y])
    x_solver=float(solve([eqw[x]-x,w-wc])[x])
    #, carla.Rotation(yaw=305, pitch=-15)
    print(x_solver)
    print(y_solver)
    print(mape.get_waypoint(carla.Location(x=x_solver/1000,y=y_solver/100000, z=1.3), project_to_road=True,lane_type=(carla.LaneType.Driving)).transform)
    return mape.get_waypoint(carla.Location(x=x_solver/1000,y=y_solver/100000, z=1.3), project_to_road=True,lane_type=(carla.LaneType.Driving))

def get_coord_normals(wc,hc, mape,max_y):
    pixels=normalizedpixel(wc,hc)
    print(pixels)
    point=convertPoint(mt,pixels[0],pixels[1])
    print(point)
    #mape.get_waypoint(carla.Location(x=point[0],y=point[1], z=1.3), project_to_road=True,lane_type=(carla.LaneType.Driving))
    return  SpecialTransform(carla.Transform(carla.Location(x=point[0],y=point[1], z=0.1)))


def get_distance(x1,x2,y1,y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)


def get_distance_loc(x1,x2):
    return sqrt((x2.location.x-x1.location.x)**2+(x2.location.y-x1.location.y)**2)
def get_direction_near(spawn_points,coords):

    x=coords.location.x
    y=coords.location.y
    minElement=spawn_points[0]
    mindist=get_distance(spawn_points[0].transform.location.x,x,spawn_points[0].transform.location.y,y)
    for i in spawn_points:
        mindisttemp=get_distance(i.transform.location.x,x,i.transform.location.y,y)
        if(mindisttemp<mindist):
            minElement=i
            mindist=mindisttemp
    
    return minElement

def draw_waypoints(waypoints, road_id=None, life_time=50.0):
    for waypoint in waypoints:
        if(waypoint.road_id == road_id):
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                   persistent_lines=True)

def sortedTime(e):
    return e[0]

def getDirection(m):
    direction=angle
    c=1
    directions=[angle,angle+180,angle+90,angle-90]
    diff=[0,0,0,0]
    while(len(m)>c and c<5):
        f=m[c-1]
        s=m[c]
        
        if(s[0]>f[0]):
            diff[0]+=abs(s[0]>f[0])
        elif(s[0]<f[0]):
            diff[1]+=abs(s[0]>f[0])
        elif(s[1]<f[1]):
            diff[2]+=abs(s[1]>f[1])
        elif(s[1]>f[1]):
            diff[3]+=abs(s[1]>f[1])
        c=c+1
    
    return directions[diff.index(max(diff))]

def main():
    
    
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    df = pd.read_csv (r'traffic_measurement.csv')
    df_locations = pd.read_csv (r'traffic_measurementlocations.csv')
    vehicles_list=[]
    
    world = client.get_world()
    settings = world.get_settings()
    settings = world.get_settings()
    #settings.fixed_delta_seconds = 1/fps
    settings.synchronousMode=True
    world.apply_settings(settings)
    
    spectator = world.get_spectator()
    blueprints = world.get_blueprint_library().filter('vehicle')
    blueprintsb = world.get_blueprint_library().filter('vehicle.bh.crossbike')
    blueprintsw = world.get_blueprint_library().filter('walker.pedestrian')

    print("Configure2")
    tm_port=8000
    traffic_manager = client.get_trafficmanager(tm_port)
    synchronous_master = True
    
        
    port = traffic_manager.get_port()
    world.get_spectator().set_transform(point_camera)
    mape=world.get_map()
    waypoint_list=world.get_map().generate_waypoints(distance=0.1)
    max_y=-20000
    for w in waypoint_list:
        if(w.transform.location.y>max_y):
            max_y=w.transform.location.y
    #for w in waypoint_list:
       # world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                     #  color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                     #  persistent_lines=True)

    
    for i in points_carla:
        world.debug.draw_string( SpecialTransform(carla.Transform(carla.Location(x=i[0],y=i[1], z=0.1))).transform.location, str(i[0])+","+str(i[1]), draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    world.debug.draw_string(get_coord_normals(714,311,mape,max_y).transform.location, "o 750,261", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    world.debug.draw_string(get_coord_normals(334,888,mape,max_y).transform.location, "o 444,984", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    world.debug.draw_string(get_coord_normals(1251,304,mape,max_y).transform.location, "z 1147,228", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    world.debug.draw_string(get_coord_normals(1670,922,mape,max_y).transform.location, "z 1123,984", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)

    world.debug.draw_string(get_coord_normals(712,308,mape,max_y).transform.location, "t 712,308", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    print(get_coord_normals(712,308,mape,max_y).transform)


    world.debug.draw_string(get_coord_normals(253,870,mape,max_y).transform.location, "t 253,870", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    print(get_coord_normals(253,870,mape,max_y).transform)



    world.debug.draw_string(get_coord_normals(1215,314,mape,max_y).transform.location, "t 1215,314", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    print(get_coord_normals(1215,314,mape,max_y).transform)

    world.debug.draw_string(get_coord_normals(1665,926,mape,max_y).transform.location, "t 1665,926", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    print(get_coord_normals(1665,926,mape,max_y).transform)


    world.debug.draw_string(get_coord_normals(1027,358,mape,max_y).transform.location, "X", draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    print(get_coord_normals(1027,358,mape,max_y).transform)




    
    import matplotlib.pyplot as plt

    plt.plot(200)
    plt.gca().invert_yaxis()
    plt.margins(x=0.7,y=0)
    plt.plot(
        [wp.transform.location.x for wp in waypoint_list],
        [wp.transform.location.y for wp in waypoint_list],
        linestyle="", markersize=3, color="blue", marker="o"
        )
    #plt.show()
    print("Configure3")
    print(len(waypoint_list))
    SetAutopilot = carla.command.SetAutopilot
    
    traffic_manager.global_percentage_speed_difference(30)
    traffic_manager.set_random_device_seed(5)
    
    try:
        objects=[]
        controllers=[]
        vehicles_carla=[]
        objects_locations=[]
        indexes_vehicles=[]
        print("Configure6")
        
        print("Configure4")
        for index, i in df.iterrows():
            vehicles_carla.append(None)
            controllers.append(None)
            indexes_vehicles.append(0)
            objects.append({
            "id":int(i["id"]),
            "type":i["Vehicle Type/Size"],
            "color":i["Vehicle Color"],
            "locations":[],
            "locations_carla":[],
            "locations_final":[]
        })
        print("Configure5")
        for index, i in df_locations.iterrows():
            print(index)
            coor= [int(i["x"]), int(i["y"]),i["time"]]
            objects[int(i["id"])]["locations_carla"].append(coor)
        objects_locations=[]
        for i in objects:
            elements=[]
            time_p=0
            sum_x=0
            sum_y=0
            count=0
            for j in i["locations_carla"]:
                if(count==num_prom):
                    elements.append([time_p,i["id"],[sum_x/count,sum_y/count]])
                    sum_x=0
                    sum_y=0
                    count=0
                else:
                    if(count==0):
                        time_p=j[2]
                    sum_x+=j[0]
                    sum_y+=j[1]
                    count+=1
            if(count!=0):
                elements.append([time_p,i["id"],[sum_x/count,sum_y/count]])
            objects[int(i["id"])]["locations_final"]=elements
            for k in elements:
                objects_locations.append(k)
        
        prev=0
        print("Configure5.1")
        objects_locations.sort(key=sortedTime)
        print("Created se")
        #blueprintw = random.choice(blueprintsw)
        #vehicle=world.try_spawn_actor(blueprintw, carla.Transform(carla.Location(x=-30.895,y=12.698, z=1.3), carla.Rotation(yaw=-150, pitch=0)))
        
        
        
        for i in objects:
            r = randrange(256)
            b = randrange(256)
            g = randrange(256)
            #for w in objects[i["id"]]["locations_carla"]:
            #    target_ini=w
            #    world.debug.draw_string(get_coord_normals(target_ini[0],target_ini[1],mape,max_y).transform.location, str(i["id"]), draw_shadow=False,
            #                           color=carla.Color(r=r, g=g, b=b), life_time=120.0,
            #                           persistent_lines=True)
        r = randrange(256)
        b = randrange(256)
        g = randrange(256)
        for i in objects_locations:
            batch = []
                    
            if(i[1]>-1):
                print("Configure7")
                
                print(i)
                if(i[0]-prev>0):
                    time.sleep(i[0]-prev)

                if(controllers[int(i[1])]==None):
                    
                    
                    target_ini=i[2]
                    print(len(target_ini))
                    target_ini=get_coord_normals(target_ini[0],target_ini[1],mape,max_y).transform
                    direction=getDirection(objects[int(i[1])]["locations_final"])
                    
                    target_ini.rotation.yaw=direction
                    target_ini.location.z=altura
                    blueprint = random.choice(blueprints)
                    while(int(blueprint.get_attribute('number_of_wheels'))!=4):
                        blueprint = random.choice(blueprints)
                    
                    if(objects[int(i[1])]["type"]!="car"):
                        blueprint = random.choice(blueprintsb)
                    if blueprint.has_attribute('color'):
                        color2=colors.to_rgb(colors.cnames[objects[int(i[1])]["color"]])
                        color2=','.join(str(x*255) for x in color2)
                        blueprint.set_attribute('color',  color2)
                    
                    print(str(target_ini))
                    world.debug.draw_string(target_ini.location, str(i[1]), draw_shadow=False,color=carla.Color(r=r, g=g, b=b), life_time=120.0,persistent_lines=True)
                    
                    vehicle=world.try_spawn_actor(blueprint, target_ini)
                    if vehicle is not None:
                        print("create vehicle")
                        custom_controller = VehiclePIDController(vehicle, args_lateral = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': 1},
        args_longitudinal = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': 1})
                        controllers[int(i[1])]=custom_controller
                        vehicles_carla[int(i[1])]=vehicle
                else:
                    print("8-prev")
                    print(objects[int(i[1])]["locations_final"])
                    print(indexes_vehicles[int(i[1])])
                    target_waypoint=objects[int(i[1])]["locations_final"][indexes_vehicles[int(i[1])]]
                    print(target_waypoint)
                    target_waypoint=get_coord_normals(target_waypoint[0],target_waypoint[1],mape,max_y)
                    target_waypoint_prev=objects[int(i[1])]["locations_final"][indexes_vehicles[int(i[1])]-1]
                    target_waypoint_prev=get_coord_normals(target_waypoint_prev[0],target_waypoint_prev[1],mape,max_y)
                    timep=0

                    #while(get_distance_loc(vehicles_carla[int(i[1])].get_transform(),target_waypoint_prev.transform)>15 and timep<50):
                        #print("No ha llegado")
                        #print(vehicles_carla[int(i[1])].get_transform().location)
                        #print(target_waypoint_prev.transform.location)
                        #control_signal = controllers[int(i[1])].run_step(1, target_waypoint_prev)
                        #vehicles_carla[int(i[1])].apply_control(control_signal)
                        #timep=timep+1
                    world.debug.draw_string(target_waypoint_prev.transform.location, str(i[1]), draw_shadow=False,color=carla.Color(r=r, g=g, b=b), life_time=120.0,persistent_lines=True)
                    world.debug.draw_string(target_waypoint.transform.location, str(i[1]), draw_shadow=False,color=carla.Color(r=r, g=g, b=b), life_time=120.0,persistent_lines=True)
                    if(get_distance_loc(vehicles_carla[int(i[1])].get_transform(),target_waypoint_prev.transform)>0.5):
                        print("Configure8")
                        target_waypoint=SpecialTransform(carla.Transform(carla.Location(x=target_waypoint.transform.location.x,y=target_waypoint.transform.location.y, z=vehicles_carla[int(i[1])].get_transform().location.z)))
                        print(target_waypoint.transform)
                        control_signal = controllers[int(i[1])].run_step(1, target_waypoint)
                        print(control_signal)
                        vehicles_carla[int(i[1])].apply_control(control_signal)
                indexes_vehicles[int(i[1])]+=1
                print("Prev")
                print(indexes_vehicles[int(i[1])])
                print(len(objects[int(i[1])]["locations_final"]))
                if(vehicles_carla[int(i[1])]!=None and indexes_vehicles[int(i[1])]>=len(objects[int(i[1])]["locations_final"])):
                    print("Set autopilot")
                    batch.append(SetAutopilot(vehicles_carla[int(i[1])], True))
                    #batch.append(Auto_lane_change(vehicles_carla[int(i[1])], True))
                    traffic_manager.auto_lane_change(vehicles_carla[int(i[1])], False)
                    
                    client.apply_batch_sync(batch)
                    
                    time.sleep(3)
            prev=i[0]
        print("finish")
        while True:
            time.sleep(5)
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        
    finally:
        print('\ndestroying %d vehicles' % len(vehicles_carla))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_carla])
        time.sleep(0.5)


if __name__ == '__main__':

    main()
