#defines classes to be used alongside the CSAPR2 labeling app
from shapely.ops import unary_union, polygonize
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from matplotlib.path import Path
import numpy as np
r_bound = 1100


class PolyMask:
    
    #defines the PolyMask object for representing user defined labels for CSAPR2 data
    #
    #   GLOBALS:
    #   mask: list of Polygons - a list of shapely polygons representing segmentation labels
    #
    #
    #   METHODS:
    #   
    #   add( vertices or x , y=None ):
    #   adds a shapely polygon to the mask from either a list of vertices stored
    #   as tuples or an x and y vector
    #
    #   pop():
    #   removes the last polygon from the mask
    #
    #   delete_point(x,y):
    #   deletes all polygons containing the point x,y
    #
    #   union():
    #   combines overlapping polygons in the mask into single polygons
    #
    #   polar_tform():
    #   returns the mask as a list of polygons transformed to polar coordinates
    #
    #   polar_raster():
    #   returns a raster representation of the mask in polar coordinates
    #
    #   as_paths():
    #   returns the mask as a list of matplotlib path objects for easy plotting
    
    def __init__(self):
        self.mask = []
    
    def add(self,verts,y=None):
        if y is None:
            self.mask.append(Polygon(verts))
        else:
            x,y = list(verts),list(y)
            self.mask.append(Polygon(list(zip(x,y))))
            
    def pop(self):
        self.mask.pop()
            
    def delete_point(self,x,y):
        self.mask = list(filter(lambda p: not p.contains(Point((x,y))),self.mask))
        
    def union(self):
        if len(self.mask) > 1:
            new_mask = unary_union(self.mask)
            if new_mask.type == 'Polygon':
                self.mask = [new_mask]
            else:
                self.mask = list(new_mask)
        
    def polar_tform(self):
        
        #functions to do coordinate transforms:
        def pol_to_cart(theta,r):
            x = r*np.cos(theta*np.pi/180)
            y = r*np.sin(theta*np.pi/180)
            return x, y
        
        def cart_to_pol(x,y):
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y,x)*180/np.pi
            theta = theta + np.double(theta<0)*360
            return theta, r
        
        def add_points_to_line(p0,p1):
            #adds points to a line segment so that it will appear as a curve after
            #a transform to polar coords (lines near the origin need to be resolved better):
            new_points = []
            x1,y1 = p1[0],p1[1]
            x0,y0 = p0[0],p0[1]
            x, y = x0,y0
            d = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            zx,zy = (x1-x0)/d, (y1-y0)/d
            while np.sqrt((x-x0)**2 + (y-y0)**2)<d:
                new_points.append((x,y))
                r = np.sqrt(x**2+y**2)
                dz = 1/(0.05 + np.abs(1/(0.001+r/10)))
                x += dz*zx
                y += dz*zy
            new_points.append((x1,y1))
            return new_points
        
        def polar_tform_ring(ring):
            #performs a cartesian to polar coordinate transform on a ring defined
            #by a list of coordinates:
            coords = list(ring.coords)
            
            #first add some points to the ring to better resolve heavily distorted regions:
            hr_coords = []
            for i in range(len(coords)-1):
                hr_coords += add_points_to_line(coords[i],coords[i+1])
            
            #do the coordinate transform:
            return [cart_to_pol(*point) for point in hr_coords]
            
        def polar_tform_polygon(poly):
            #cartesian to polar transform for a shapely polygon
            exterior = polar_tform_ring(poly.exterior)
            interiors = [polar_tform_ring(ring) for ring in poly.interiors]
            return Polygon(exterior,interiors)
        
        def break_self_intersect(p):
            #polygons with self-intersecting exteriors need to be broken into multiple polygons
            ext = unary_union(LineString(p.exterior))
            broken = MultiPolygon(polygonize(ext))
            if len(p.interiors)>0:
                holes = []
                for i in p.interiors:
                    holes.append(Polygon(i))
                holes = MultiPolygon(holes)
                broken = broken.difference(holes)
            return list(broken)
            
        # #make sure that the polygons are not self-intersecting:
        polygons = self.mask
        simple_polygons = []
        for p in polygons:
            if p.is_simple:
                simple_polygons.append(p)
            else:
                simple_polygons.extend(break_self_intersect(p))
                
        #combine polygons:
        mask = unary_union(simple_polygons)
        
        #handle the branch cut:
        eps = 0.0001
        r = np.array([r_bound,eps,eps,eps,eps,eps,r_bound])
        theta = np.array([eps,eps,90,180,270,360-eps,360-eps])
        x, y = pol_to_cart(theta,r)
        branch_cut = Polygon(list(zip(x,y)))
        mask = mask.difference(branch_cut)
        if mask.type=='Polygon':
            mask = [mask]
        else:
            mask = list(mask)
        
        #transform each polygon:
        polar_polys = [polar_tform_polygon(p) for p in mask]
        
        return polar_polys
    
    def polar_raster(self):
        
        def rasterize(mask,x,y):
        
            #this is a recursive function to convert a multipolygon mask into a raster mask:
            #rather than checking each point in the raster it recursively subdivides the raster
            #into into smaller rectangles and checks whether they are disjoint or within the mask,
            #and when they are, labels all of the raster coordinates in the rectangle at once
            
            def contains_points(mask,x,y):
                #checks if the points in 'x,y' are contained in the multipolygon 'mask' this
                #will be called when a rectangle is small and contains only a handful of points
                raster = np.zeros((x.size,y.size))
                for i in range(x.size):
                    for j in range(y.size):
                        raster[i,j] = Point((x[i],y[j])).within(mask)  
                return raster
            
            #if one of the inputs is empty there is nothing to check
            if x.size==0 or y.size==0:
                return np.zeros((x.size,y.size))
            
            #if one of the input dimensions is 1 just check the remaining points individually
            if x.size == 1 or y.size == 1:
                return contains_points(mask,x,y)
            
            #a polygon representing the input region
            box = Polygon([(x[0],y[0]),(x[0],y[-1]),(x[-1],y[-1]),(x[-1],y[0])])
            
            #if the input box is disjoint from the mask fill the box with zeros
            if mask.disjoint(box):
                return np.zeros((x.size,y.size))
            
            #if the input box is entirely contained in the mask fill with ones
            if box.within(mask):
                return np.ones((x.size,y.size))
            
            #if the box and the mask partially intersect the box needs to be broken up:
            northwest = rasterize(mask,x[:x.size//2],y[:y.size//2])
            northeast = rasterize(mask,x[x.size//2:],y[:y.size//2])
            southwest = rasterize(mask,x[:x.size//2],y[y.size//2:])
            southeast = rasterize(mask,x[x.size//2:],y[y.size//2:])
            
            return np.concatenate(
                (np.concatenate((northwest,northeast),axis=0),
                 np.concatenate((southwest,southeast),axis=0)),axis=1)
        
        theta = np.arange(360)
        r = np.arange(r_bound)
        mask = MultiPolygon(self.polar_tform())
        return rasterize(mask,theta,r)
    
    def as_paths(self):
        
        #returns the polygon mask as a list of paths for plotting with matplotlib
        def generate_codes(n):
            return [Path.MOVETO] + [Path.LINETO] * (n - 1)

        def poly_to_path(poly):
            vertices = list(poly.exterior.coords)
            codes = generate_codes(len(poly.exterior.coords))
            for interior in poly.interiors:
                vertices.extend(interior.coords)
                codes.extend(generate_codes(len(interior.coords)))
            return Path(vertices, codes)
        
        return [poly_to_path(p) for p in self.mask]

#testing code:
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = PolyMask()
    p.add([(0,0),(0,1000),(100,1000),(100,100),(1000,100),(1000,0)])
    p.add([(0,900),(0,1000),(1000,0),(900,0)])
    p.add(((-800,-50),(-800,-150),(800,-150),(800,-50)))
    p.add(((-50,-500),(-50,-550),(50,-550),(50,-500)))
    p.add(((-10,-10),(-10,10),(-1000,10),(-1000,-10)))
    raster = p.polar_raster()
    plt.pcolormesh(raster.T)