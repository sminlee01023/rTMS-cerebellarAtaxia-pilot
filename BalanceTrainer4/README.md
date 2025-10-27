### Preprocessing and Feature Extraction for Balance Trainer 4(HUR, Finland) data
  <br />
The Balance Trainer 4 (HUR, Finland) is a portable balance assessment platform designed for both clinical and field environments.  <br />
It supports a variety of standardized balance test protocols and interactive exercises using the included HUR SmartBalance software, providing clear visual feedback, motivational training tasks, and detailed reporting.  <br />
The system offers portability, operates without external power, and enables comparison with normative data for versatile balance evaluation.
  <br />
  <br />
For this project, balance data were collected from patients with cerebellar ataxia (CA) using the Balance Trainer 4 (HUR, Finland).  <br />
Each participant was asked to maintain a standing position for 30 seconds under two conditions: with eyes closed and with eyes open.  <br />
These measurements provide quantitative assessment of postural stability in CA patients.
  <br />
  <br />
The datasets from this device cannot be shared publicly due to data privacy restrictions.

  <br />
  <br />
  
#### BT4 Data Adjustment and Confidence Ellipse Visualization
Since the first coordinate in the Balance Trainer 4 (BT4) data is not at the origin (0, 0), the data is first adjusted by translating all coordinates so that the initial point aligns with (0, 0). This normalization facilitates meaningful spatial comparison across trials and subjects.

Because the balance data are two-dimensional (X and Y positions), assessing variability and confidence intervals is performed using confidence ellipses. The confidence ellipse visually represents the region where the true mean values lie with a specified probability (e.g., 90%, 95%). This statistical approach accounts for covariance between the two spatial dimensions, providing a comprehensive view of postural stability.


  <br />
  <br />
  
#### Code Usage
```python
from zero_adj import zero_adj
from ellipse_plot import organize_and_plot_ellipse
import pandas as pd
import glob

base_dir = 'your/base/directory/'
categories = ['01_PRE', '02_POST', '03_FOLLOW']
adj_dir = 'your/adj/directory/'

raw_dirs = sorted(glob.glob(base_dir+'*.csv')
for raw_dir in raw_dirs:
  zero_adj(pd.read_csv(raw_dir, index_col = None), save_dir=adj_dir + 'your/file/name.csv')

organize_and_plot_ellipse(adj_dir, categories)

```
