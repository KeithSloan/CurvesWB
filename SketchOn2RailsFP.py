# -*- coding: utf-8 -*-

__title__ = "Sketch on 2 rails"
__author__ = "Christophe Grellier (Chris_G)"
__license__ = "LGPL 2.1"
__doc__ = """Sketch normal to an edge, with up vector normal to a face"""

import sys
if sys.version_info.major >= 3:
    from importlib import reload

import FreeCAD
import FreeCADGui
import Part
import Sketcher
import _utils

TOOL_ICON = _utils.iconsPath() + '/sketch_on_2_rails.svg'
#debug = _utils.debug
#debug = _utils.doNothing

props = """
App::PropertyBool
App::PropertyBoolList
App::PropertyFloat
App::PropertyFloatList
App::PropertyFloatConstraint
App::PropertyQuantity
App::PropertyQuantityConstraint
App::PropertyAngle
App::PropertyDistance
App::PropertyLength
App::PropertySpeed
App::PropertyAcceleration
App::PropertyForce
App::PropertyPressure
App::PropertyInteger
App::PropertyIntegerConstraint
App::PropertyPercent
App::PropertyEnumeration
App::PropertyIntegerList
App::PropertyIntegerSet
App::PropertyMap
App::PropertyString
App::PropertyUUID
App::PropertyFont
App::PropertyStringList
App::PropertyLink
App::PropertyLinkSub
App::PropertyLinkList
App::PropertyLinkSubList
App::PropertyMatrix
App::PropertyVector
App::PropertyVectorList
App::PropertyPlacement
App::PropertyPlacementLink
App::PropertyColor
App::PropertyColorList
App::PropertyMaterial
App::PropertyPath
App::PropertyFile
App::PropertyFileIncluded
App::PropertyPythonObject
Part::PropertyPartShape
Part::PropertyGeometryList
Part::PropertyShapeHistory
Part::PropertyFilletEdges
Sketcher::PropertyConstraintList
"""

class SketchOn2RailsFP:
    """Creates a Oriented sketch"""
    def __init__(self, obj, e1, e2):
        """Add the properties"""
        obj.addProperty("App::PropertyLinkSub", "Rail1", "SketchOn2Rails", "First rail")
        obj.addProperty("App::PropertyLinkSub", "Rail2", "SketchOn2Rails", "Second rail")
        obj.addProperty("App::PropertyFloatConstraint", "Parameter", "SketchOn2Rails", "Parameter on edge")
        obj.addProperty("App::PropertyFloat", "CrossLength", "SketchOn2Rails", "Distance between the 2 rails")
        obj.addProperty("App::PropertyEnumeration", "Normal", "SketchOn2Rails", "Sketch normal").Normal = ["Rail1","Rail2"]
        obj.Proxy = self
        obj.Normal = "Rail1"
        obj.Rail1 = e1
        obj.Rail2 = e2
        obj.Parameter = (0.0, 0.0, 1.0, 0.05)

    def execute(self, obj):
        edges = []
        for g in obj.Geometry:
            if hasattr(g, 'Construction') and not g.Construction:
                #try:
                edges.append(g.toShape())
                #except AttributeError:
                    #debug("Failed to convert %s to BSpline"%str(g))
        if edges:
            c = Part.Compound([])
            se = Part.sortEdges(edges)
            for l in se:
                if len(l) > 1:
                    c.add(Part.Wire(l))
                elif len(l) == 1:
                    c.add(l[0])
            obj.Shape = c

    def onChanged(self, obj, prop):
        if prop == "Parameter":
            e1 = _utils.getShape(obj, "Rail1", "Edge")
            p1 = e1.FirstParameter + (e1.LastParameter - e1.FirstParameter) * obj.Parameter
            loc1 = e1.valueAt(p1)
                        
            e2 = _utils.getShape(obj, "Rail2", "Edge")
            p2 = e2.FirstParameter + (e2.LastParameter - e2.FirstParameter) * obj.Parameter
            loc2 = e2.valueAt(p2)
            
            # obj.Normal == "Rail1"
            z = e1.tangentAt(p1)
            x = loc2-loc1
            loc = loc1
            
            if obj.Normal == "Rail2":
                z = e2.tangentAt(p2)
                x = loc1-loc2
                loc = loc2
            #elif obj.Normal == "Average":
                #z = e1.tangentAt(p1)
                #x = loc2-loc1
                #loc = loc1
                
            obj.CrossLength = x.Length
            #print("{0!s} - {1!s}".format(p, loc))
            y = z.cross(x)
            rot = FreeCAD.Rotation(x, y, z, "XZY")
            obj.Placement.Base = loc
            obj.Placement.Rotation = rot
            obj.solve()

        if prop == "CrossLength":
            obj.solve()


class SketchOn2RailsVP:
    def __init__(self, vobj):
        vobj.Proxy = self

    def getIcon(self):
        return TOOL_ICON

    def attach(self, vobj):
        self.Object = vobj.Object

    def __getstate__(self):
        return {"name": self.Object.Name}

    def __setstate__(self, state):
        self.Object = FreeCAD.ActiveDocument.getObject(state["name"])
        return None

class sketch_on_2_rails_cmd:
    """Creates a Oriented sketch"""
    def makeFeature(self, e1, e2):
        fp = FreeCAD.ActiveDocument.addObject("Sketcher::SketchObjectPython", "")
        SketchOn2RailsFP(fp, e1, e2)
        SketchOn2RailsVP(fp.ViewObject)
        FreeCAD.ActiveDocument.recompute()
        fp.addGeometry(Part.LineSegment(FreeCAD.Vector(0, 0, 0),FreeCAD.Vector(1, 0, 0)),True)
        fp.addConstraint(Sketcher.Constraint('Coincident',0,1,-1,1)) 
        fp.addConstraint(Sketcher.Constraint('PointOnObject',0,2,-1)) 
        fp.addConstraint(Sketcher.Constraint('DistanceX',-1,1,0,2,1.0)) 
        fp.setExpression('Constraints[2]', u'CrossLength')
        FreeCAD.ActiveDocument.recompute()

    def Activated(self):
        sel = FreeCADGui.Selection.getSelectionEx()
        e = []
        for s in sel:
            for sen in s.SubElementNames:
                if "Edge" in sen:
                    e.append((s.Object, sen))
        if len(e) >= 2:
            self.makeFeature(e[0], e[1])
        else:
            FreeCAD.Console.PrintError("Select 2 edges.\n")

    def IsActive(self):
        if FreeCAD.ActiveDocument:
            return True
        else:
            return False

    def GetResources(self):
        return {'Pixmap' : TOOL_ICON, 'MenuText': __title__, 'ToolTip': __doc__}

FreeCADGui.addCommand('sketch_on_2_rails', sketch_on_2_rails_cmd())
