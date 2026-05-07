from crest.model import IOSpec,TensorSpec

def test_iospecs():
    ios = IOSpec()
    keys = ['cape', 'cp', 'csfr'] 
    ios['ERA5'] = {'keys' : keys, 
                   'coord_shapes' :  {k :{'datetime' : None, 'latitude' : 1, 'longitude' : 1} for k in keys}
                   }
    
    keys = ['Actual', 'Equipped', 'GroundWater','Non-Conventional', 'SurfaceWater']
    ios['Irrigation'] = {'keys' : keys,
                         'coord_shapes': {k : {'latitude' : 1, 'longitude' : 1} for k in keys}
    }
                        
    ios2 = IOSpec()
    ios2['ERA5'] = {'keys' : ['cape', 'cp', 'csfr'] , 
                   'coord_shapes' :  {'datetime' : None, 'latitude' : 1, 'longitude' : 1}
                   }
    
    ios2['Irrigation'] = {'keys' : ['Actual', 'Equipped', 'GroundWater','Non-Conventional', 'SurfaceWater'],
                         'coord_shapes': {'latitude' : 1, 'longitude' : 1} 
                         }
    
    for k in ios.spec:
        assert ios2.spec[k].specs == ios.spec[k].specs

    encode = ios.encode()
    decode = ios.decode(encode)
    for k in ios.spec:
        assert ios.spec[k].specs == decode.spec[k].specs