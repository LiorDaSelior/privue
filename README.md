# privue
Package for Local Differential Privacy data privatization and estimation using the privatized data, following the Unary Encoding technique.  
Install the package: `pip install privue`  

### Stremlit Demo
Try out a demo to better understand the privatization and estimation: https://privue-demo-wdxg3n5hvmu2gcv8ypw432.streamlit.app/  
Demo page github repo: https://github.com/LiorDaSelior/privue-demo  

### Documentation
https://liordaselior.github.io/privue/  

## Overview
Many software companies prioritize collection and analysis of data from users' devices.  
While data collection improves the user experience, it may harm users by leaking their sensitive information.  
Differential Privacy (DP) is a powerful mechanism for data perturbation, widely adopted by numerous companies, which provides privacy guarantees when collecting and analyzing user data.  
Local Differential Privacy (LDP) is a stronger variant that, in addition to providing the same privacy guarantees, also privatizes the data before it leaves the user device.  
With this package, we aim to increase awareness for privatized data collection, by providing an accessible tool for data perturbation using the Symmetric Unary Encoding (SUE) algorithm.  
Please read the report (in the `report/` folder) for a more in-depth review.  
The package provides 3 relevant subpackages:  
-  privue.client - includes functions for LDP data perturbation on the user's side (before user submits the data).  
-  privue.server - includes functions for estimation of the aggregated privatized data (after collecting it from each user).  
-  privue.util - includes functions for `.json` format support (which you can try out in the Streamlit demo).  


