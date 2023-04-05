#!/bin/bash
set -ex

# Install hugging face with GPU support if that is what makes sense on the machine.
pip install -r requirements.txt
(pip install 'transformers[torch]' &&
  python -c "import sys,torch; use_cuda = torch.cuda.is_available(); sys.exit(not use_cuda)") \
    || pip install 'transformers[tf-cpu]'

# Run the sample, which will incidentally download the model.
python sample.py
