classdef TransformSemantics < SwigRef
  methods
    function self = TransformSemantics(varargin)
      if nargin~=1 || ~ischar(varargin{1}) || ~strcmp(varargin{1},'_swigCreate')
        % How to get working on C side? Commented out, replaed by hack below
        %self.swigCPtr = iDynTreeMATLAB_wrap(168,'new_TransformSemantics',varargin{:});
        %self.swigOwn = true;
        tmp = iDynTreeMATLAB_wrap(168,'new_TransformSemantics',varargin{:}); % FIXME
        self.swigCPtr = tmp.swigCPtr;
        self.swigOwn = tmp.swigOwn;
        self.swigType = tmp.swigType;
        tmp.swigOwn = false;
      end
    end
    function delete(self)
      if self.swigOwn
        iDynTreeMATLAB_wrap(169,'delete_TransformSemantics',self);
        self.swigOwn=false;
      end
    end
    function varargout = getRotationSemantics(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(170,'TransformSemantics_getRotationSemantics',self,varargin{:});
    end
    function varargout = getPositionSemantics(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(171,'TransformSemantics_getPositionSemantics',self,varargin{:});
    end
    function varargout = setRotationSemantics(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(172,'TransformSemantics_setRotationSemantics',self,varargin{:});
    end
    function varargout = setPositionSemantics(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(173,'TransformSemantics_setPositionSemantics',self,varargin{:});
    end
    function varargout = getPoint(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(174,'TransformSemantics_getPoint',self,varargin{:});
    end
    function varargout = getOrientationFrame(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(175,'TransformSemantics_getOrientationFrame',self,varargin{:});
    end
    function varargout = getReferencePoint(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(176,'TransformSemantics_getReferencePoint',self,varargin{:});
    end
    function varargout = getReferenceOrientationFrame(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(177,'TransformSemantics_getReferenceOrientationFrame',self,varargin{:});
    end
    function varargout = setPoint(self,varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(178,'TransformSemantics_setPoint',self,varargin{:});
    end
    function varargout = setOrientationFrame(self,varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(179,'TransformSemantics_setOrientationFrame',self,varargin{:});
    end
    function varargout = setReferencePoint(self,varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(180,'TransformSemantics_setReferencePoint',self,varargin{:});
    end
    function varargout = setReferenceOrientationFrame(self,varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(181,'TransformSemantics_setReferenceOrientationFrame',self,varargin{:});
    end
    function varargout = inverse(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(188,'TransformSemantics_inverse',self,varargin{:});
    end
    function varargout = mtimes(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(189,'TransformSemantics_mtimes',self,varargin{:});
    end
    function varargout = toString(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(190,'TransformSemantics_toString',self,varargin{:});
    end
    function varargout = display(self,varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(191,'TransformSemantics_display',self,varargin{:});
    end
    function [v,ok] = swig_fieldsref(self,i)
      v = [];
      ok = false;
      switch i
      end
    end
    function [self,ok] = swig_fieldasgn(self,i,v)
      switch i
      end
    end
  end
  methods(Static)
    function varargout = check_compose(varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(182,'TransformSemantics_check_compose',varargin{:});
    end
    function varargout = check_inverse2(varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(183,'TransformSemantics_check_inverse2',varargin{:});
    end
    function varargout = check_transform(varargin)
      [varargout{1:max(1,nargout)}] = iDynTreeMATLAB_wrap(184,'TransformSemantics_check_transform',varargin{:});
    end
    function varargout = compose(varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(185,'TransformSemantics_compose',varargin{:});
    end
    function varargout = inverse2(varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(186,'TransformSemantics_inverse2',varargin{:});
    end
    function varargout = transform(varargin)
      [varargout{1:nargout}] = iDynTreeMATLAB_wrap(187,'TransformSemantics_transform',varargin{:});
    end
  end
end
