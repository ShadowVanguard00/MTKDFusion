import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=6, help='# of threads for data loader')
    self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    self.parser.add_argument('--n_ep', type=int, default=50, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    self.parser.add_argument('--name', type=str, default='ka', help='folder name to save outputs')

   
  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

 