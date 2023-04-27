configs = {

    # --------------------------------------------------------------
    '1d' : {

        # Data definitions
        'data_kwargs' : [
            {'dimensions' : {'x' :  9}, 'chunks': {'x': 3}}, # 1st data object 
            {'dimensions' : {'x' : 17}, 'chunks': {'x': 6}}, # 2nd data object  
            {'dimensions' : {'x' :  6}, 'chunks': {'x': 2}}, # 3rd data object 
        ],

        # Window depths
        'depth' : [
            {'x': 1}, 
            {'x': 2}, 
            {'x': 0},
        ],
    },


    # --------------------------------------------------------------
    '3d' : {

        # Data definitions
        'data_kwargs' : [

            { # First Datafile
                'dimensions' : {
                    'features'  : 2,
                    'time'      : 3,
                    'latitude'  : 3,
                    'longitude' : 3,
                }, 
            },

            { # Second Datafile
                'dimensions' : { 
                    'time'      : 3,
                    'latitude'  : 5,
                    'longitude' : 5,
                }, 
            },
        ],

        # Initial data chunking
        'chunks' : { 
            'time'      : -1, 
            'latitude'  : 2,
            'longitude' : 2,
        },

        # Window depths
        'depth' : [

            { # First - 2 x 3 x 3 window
                'time'      : (1, 0), 
                'latitude'  : (1, 1), 
                'longitude' : 1,
            },

            { # Second - 2 x 3 x 3 window
                'time'      : (1, 0), 
                'latitude'  : (1, 1), 
                'longitude' : 1, 
            },
        ],
    },

    # --------------------------------------------------------------
    'static' : {

        # Data definitions
        'data_kwargs' : [

            { # First Datafile
                'dimensions' : {
                    'time'      : 3,
                    'latitude'  : 3,
                    'longitude' : 3,
                }, 
            },

            { # Second Datafile
                'dimensions' : { 
                    'latitude'  : 5,
                    'longitude' : 5,
                }, 
            },
        ],

        # Initial data chunking
        'chunks' : { 
            'latitude'  : 2,
            'longitude' : 2,
        },

        # Window depths
        'depth' : [

            { # First - 2 x 3 x 3 window
                'time'      : (1, 0), 
                'latitude'  : 1, 
                'longitude' : 1,
            },

            { # Second - 4 x 3 window
                'latitude'  : (1, 2), 
                'longitude' : 1, 
            },
        ],
    },

    # --------------------------------------------------------------
    'extra_large' : {

        # Data definitions
        'data_kwargs'   : [

            { # First Datafile
                'dimensions' : { 
                    'time'      : 10,
                    'latitude'  : 1000,
                    'longitude' : 1000,
                }, 
                'missing_pct' : 0.05,
                'random_seed' : 42,
            },

            { # Second Datafile
                'dimensions' : { 
                    'time'      : 50,
                    'latitude'  : 500,
                    'longitude' : 500,
                }, 
            },

            { # Third Datafile
                'dimensions' : { 
                    'time'      : 300,
                    'latitude'  : 50,
                    'longitude' : 50,
                }, 
            },
        ],

        # Window depths
        'depth' : [

            { # First - 2 x 5 x 5 window
                'time'      : (1, 0),
                'latitude'  : 2, 
                'longitude' : 2,
            },

            { # Second - 2 x 3 x 3 window
                'time'      : (1, 0),
                'latitude'  : 1, 
                'longitude' : 1, 
            },

            { # Third - 3 x 3 x 3 window
                'time'      : (2, 0),
                'latitude'  : 1, 
                'longitude' : 1,
            },
        ],
    },

    # --------------------------------------------------------------
    'single' : {

        # Data definitions
        'data_kwargs' : [
            {'dimensions' : {'x' : 8}, 'chunks': {'x': 3}}, # 1st data object 
        ],

        # Window depths
        'depth' : [
            {'x': 1}, 
        ],
    },
}