# -*- coding: utf-8 -*-

from mainloop_discriminator import MainLoop

class MainLoop(MainLoop):
    def __init__(self, model, logger, train_args, model_args):
        # Call parent's init first
        super(MainLoop, self).__init__(model, logger, train_args, model_args)

    def run(self):
        """Run training loop."""
        self.model.set_dropout(True)
        #self.model.save(self.model.save_path + '.npz')
        cur_loss = self.model.val_loss()
        self.__print("Loss: %f" % cur_loss)
