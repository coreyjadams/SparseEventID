import torch
import numpy

class LossCalculator(torch.nn.Module):

    def __init__(self, vertex_shape=None):

        torch.nn.Module.__init__(self)

        self.losses = ["classification"]
        if vertex_shape is not None:
            self.losses += ["vertex"]
            self.vertex_shape = vertex_shape
            self.vertex_presence_criterion = torch.nn.BCELoss()

        # These are for classification:
        self.num_classes = {
            'label_cpi' : 2,
            'label_prot' : 3,
            'label_npi' : 2,
            'label_neut' : 3,
        }
        self._criterion = torch.nn.CrossEntropyLoss()


    def set_image_dimensions(self, image_dimensions, origin):
        self.image_dimensions = image_dimensions
        self.image_origin     = origin
        self.create_anchor_boundaries()

    def create_anchor_boundaries(self):
        print("Image Dimensions: ", self.image_dimensions)
        print("Vertex Shape: ", self.vertex_shape)
        pass

    def classification_loss(self, inputs, logits):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''

        loss = None
        for key in self.num_classes:
            values, target = torch.max(inputs[key], dim=1)
            temp_loss = self._criterion(logits[key], target = target)

            if loss is None:
                loss = temp_loss
            else:
                loss += temp_loss

        return loss

    def vertex_loss(self, minibatch_data, vertex):
        '''MSE vertex loss: x/y/z distance'''

        target = torch.squeeze(minibatch_data['vertex'])

        # First, detect the index of the anchors that the vertexes use:
        classification_target = target - self.image_origin
        ratio = classification_target / self.image_dimensions
        bin = ratio * self.vertex_shape
        bin = tuple(bin.long())
        # Pull off the classification component of the vertex:
        # It's the first channel
        vertex_class = vertex[:,0,:,:,:]

        vertex_class_target = torch.zeros(size=vertex_class.shape, device=vertex_class.device)
        for i, b in enumerate(bin):
            index = (i,) + tuple(b)
            vertex_class_target[index] = 1

        # Now, vertex_class_target is 1 only when the anchors should be positive.

        # pull off the classification target:
        vertex_prediction = torch.sigmoid(vertex_class)
        prediction_loss = self.vertex_presence_criterion(vertex_prediction, vertex_class_target)
        print(prediction_loss)

        # Now, if the 

        # Compute the vertex loss

        # The vertex prediction comes out between 0 and 1
        # for each dimension.  Map it to the real coordinates:
        vertex = vertex * self.image_dimensions


        vertex_loss = torch.nn.functional.mse_loss(target, vertex)
        # exit()

        return 0.001*vertex_loss

    def forward(self, minibatch_data, predictions):

        loss = {}

        if "classification" in self.losses:

            inputs = { key : minibatch_data[key] for key in self.num_classes }

            loss["classification"] = self.classification_loss(inputs, predictions["logits"])

        if "vertex" in self.losses:

            loss = self.vertex_loss(minibatch_data, predictions["vertex"])

        return loss
