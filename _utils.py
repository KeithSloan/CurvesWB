# -*- coding: utf-8 -*-

__title__ = "Curves workbench utilities"
__author__ = "Christophe Grellier (Chris_G)"
__license__ = "LGPL 2.1"
__doc__ = "Curves workbench utilities common to all tools."

import FreeCAD

def setIconsPath(path):
    global icons_path
    icons_path = path
    return(True)

def iconsPath():
    global icons_path
    return(icons_path)

def debug(string):
    FreeCAD.Console.PrintMessage("%s\n"%string)

def doNothing(string):
    return(None)

def getSubShape(shape, shape_type, n):
    if shape_type == "Vertex" and len(shape.Vertexes) >= n:
        return(shape.Vertexes[n-1])
    elif shape_type == "Edge" and len(shape.Edges) >= n:
        return(shape.Edges[n-1])
    elif shape_type == "Face" and len(shape.Faces) >= n:
        return(shape.Faces[n-1])
    else:
        return(None)

def getShape(obj, prop, shape_type):
    if hasattr(obj, prop):
        if obj.getTypeIdOfProperty(prop) == "App::PropertyLinkSub":
            n = eval(obj.getPropertyByName(prop)[1][0].lstrip(shape_type))
            return(getSubShape(obj.getPropertyByName(prop)[0].Shape, shape_type, n))
        elif obj.getTypeIdOfProperty(prop) == "App::PropertyLinkSubList":
            res = []
            for tup in obj.getPropertyByName(prop):
                n = eval(tup[1][0].lstrip(shape_type))
                res.append(getSubShape(tup[0].Shape, shape_type, n))
            return(res)
        else:
            FreeCAD.Console.PrintError("CurvesWB._utils.getShape: wrong property type.\n")
            return(None)
    else:
        FreeCAD.Console.PrintError("CurvesWB._utils.getShape: %r has no property %r\n"%(obj, prop))
        return(None)



class EasyProxy(object):
    def __init__(self, fp):
        self.document_restored = True
        self.ep_add_properties(fp)
        fp.Proxy = self
        self.ep_init(fp)

    def execute(self, fp):
        if not self.document_restored:
            debug("Skipping %s.execute() ..."%fp.Label)
            return(False)
        else:
            self.ep_execute(fp)

    def onChanged(self, fp, prop):
        if not self.document_restored:
            debug("Skipping %s.onChanged(%s) ..."%(fp.Label,prop))
            return(False)
        else:
            self.ep_prop_changed(fp, prop)

    def onBeforeChange(self, fp, prop):
        if prop == "Proxy":
            return(False)
        if not self.document_restored:
            debug("Skipping %s.onBeforeChange(%s) ..."%(fp.Label,prop))
            return(False)
        else:
            self.ep_before_prop_change(fp, prop)

    def onDocumentRestored(self, fp):
        self.document_restored = True
        debug("%s restored !"%fp.Label)
        self.ep_init(fp)

    def __getstate__(self):
        debug("EasyProxy.__getstate__")
        state = self.ep_on_save()
        # add additional instance variables
        # state["variable"] = self.variable
        return(state)

    def __setstate__(self,state):
        debug("EasyProxy.__setstate__")
        self.document_restored = False
        self.ep_on_restore(state)
        # restore additional instance variables
        # self.variable = state["variable"]
        return(None)

    def ep_add_properties(self, fp):
        #fp.addProperty("App::PropertyInteger", "myprop", "Test", "a property").myprop = 1
        return(None)

    def ep_init(self, fp):
        return(None)

    def ep_execute(self, fp):
        return(None)

    def ep_prop_changed(self, fp, prop):
        return(None)

    def ep_before_prop_change(self, fp, prop):
        return(None)

    def ep_on_save(self):
        return(dict())

    def ep_on_restore(self, state):
        return(None)



class MyProxy(EasyProxy):

    def ep_add_properties(self, fp):
        debug("---MyProxy.ep_add_properties")
        fp.addProperty("App::PropertyInteger", "mprop", "Test", "a property").mprop = 1
        fp.addProperty("App::PropertyInteger", "aprop", "Test", "a property").aprop = 2
        fp.addProperty("App::PropertyInteger", "zprop", "Test", "a property").zprop = 3

    def ep_init(self, fp):
        debug("---MyProxy.ep_init(%s)"%fp.Label)

    def ep_execute(self, fp):
        debug("---MyProxy.ep_execute(%s)"%fp.Label)

    def ep_prop_changed(self, fp, prop):
        debug("---MyProxy.ep_prop_changed: %s(%s)"%(fp.Label,prop))

    def ep_before_prop_change(self, fp, prop):
        debug("---MyProxy.ep_before_prop_change: %s(%s)"%(fp.Label,prop))

    def ep_on_save(self):
        debug("---MyProxy.ep_on_save")
        return(None)

    def ep_on_restore(self,state):
        debug("---MyProxy.ep_on_restore")
        return(None)
