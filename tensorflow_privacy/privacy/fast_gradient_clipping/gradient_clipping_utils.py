# Copyright 2022, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions that help in the computation of per-example gradient norms."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

from absl import logging
import tensorflow as tf

from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry as lr
from tensorflow_privacy.privacy.fast_gradient_clipping import tensorflow_graph_utils as tgu

PackedTensors = Union[tf.Tensor, Iterable[tf.Tensor], Dict[Text, tf.Tensor]]

GeneratorFunction = Callable[[Any, Tuple, Dict], Tuple[Any, Any]]


def has_internal_compute_graph(input_object: Any):
  """Checks if input is a TF model and has a TF internal compute graph."""
  return (
      isinstance(input_object, tf.keras.Model)
      and hasattr(input_object, '_flatten_to_reference_inputs')
      and hasattr(input_object, '_tensor_usage_count')
      and hasattr(input_object, '_conform_to_reference_input')
      and hasattr(input_object, '_nodes_by_depth')
  )


def _get_internal_layers(
    input_layer: tf.keras.layers.Layer,
) -> List[tf.keras.layers.Layer]:
  """Returns a list of layers that are nested within a given layer."""
  internal_layers = []
  if isinstance(input_layer, tf.keras.Model) and hasattr(input_layer, 'layers'):
    for layer in input_layer.layers:
      internal_layers.extend(_get_internal_layers(layer))
  else:
    internal_layers.append(input_layer)
  return internal_layers


def model_forward_pass(
    input_model: tf.keras.Model,
    inputs: PackedTensors,
    generator_fn: Optional[GeneratorFunction] = None,
) -> Tuple[PackedTensors, List[Any]]:
  """Does a forward pass of a model and returns useful intermediates.

  NOTE: the graph traversal algorithm is an adaptation of the logic in the
    _run_internal_graph() method in the functional.Functional class. Hence,
    forward_norm_pass should only be invoked if the generated model
    instance is an instance of the functional.Functional class.

  Args:
    input_model: A `tf.keras.Model` to compute the quantities for.
    inputs: Arbitrary input to be fed into the input layer of the model. It is
      expected that `input_model(inputs)` returns a valid output.
    generator_fn: A function with signature `(tf.keras.layers.Layer, Any, Any)
      -> (tf.Tensor, Any)`, where we require `generator_fn(layer_instance, args,
      kwargs)[0] == layer_instance(*args, **kwargs)`. If `None`, then
      `layer_fn(layer_instance, args, kwargs)[1] == None`.

  Returns:
    A `tuple` `(outputs, generator_outputs_list)`. `outputs` is the
    `PackedTensor` that is generated as a result of a forward pass.
    `generator_outputs_list` is a `list` whose i-th entry is the output of
    `generator_fn(lyr, args, kwargs)[1]` where `lyr` is the i-th
    layer when the compute graph of `input_model` is traversed in BFS order.
  """
  # TODO: Avoid or remove the references to protected methods of `input_model`.  # pylint: disable=g-bad-todo

  # Default generator.
  generator_outputs_list = []
  if generator_fn is None:

    def generator_fn(layer_instance, args, kwargs):
      return layer_instance(*args, **kwargs), None

  # Prepare the inputs and BFS variables.
  flattened_inputs = input_model._flatten_to_reference_inputs(inputs)  # pylint: disable=protected-access
  tensor_dict = {}
  tensor_usage_count = input_model._tensor_usage_count  # pylint: disable=protected-access
  for x, y in zip(input_model.inputs, flattened_inputs):
    y = input_model._conform_to_reference_input(y, ref_input=x)  # pylint: disable=protected-access
    x_id = str(id(x))
    tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
  nodes_by_depth = input_model._nodes_by_depth  # pylint: disable=protected-access
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Perform BFS feedforward computations.
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      if node.is_input:
        continue  # inputs already exist
      if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
        continue  # node is not computable; try skipping
      args, kwargs = node.map_arguments(tensor_dict)
      if has_internal_compute_graph(node.layer):
        # If this node has an internal computational graph, we can recurse.
        node_layer_outputs, node_generator_outputs = model_forward_pass(
            node.layer, args, generator_fn
        )
        generator_outputs_list.extend(node_generator_outputs)
      else:
        # Otherwise, we parse the node directly.
        node_layers = _get_internal_layers(node.layer)
        for layer in node_layers:
          node_layer_outputs, layer_generator_outputs = generator_fn(
              layer, args, kwargs
          )
          generator_outputs_list.append(layer_generator_outputs)
          args = (
              node_layer_outputs
              if isinstance(node_layer_outputs, tuple)
              else (node_layer_outputs,)
          )
          kwargs = {}

      # Update the current dictionary of inputs for the next node.
      for x_id, y in zip(
          node.flat_output_ids, tf.nest.flatten(node_layer_outputs)
      ):
        tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

  # Gather outputs (in case there are multiple) and return.
  output_tensors = []
  for x in input_model.outputs:
    x_id = str(id(x))
    output_tensors.append(tensor_dict[x_id].pop())
  model_outputs = tf.nest.pack_sequence_as(
      input_model._nested_outputs,  # pylint: disable=protected-access
      output_tensors,
  )

  return model_outputs, generator_outputs_list


def all_trainable_layers_are_registered(
    input_model: tf.keras.Model, layer_registry: lr.LayerRegistry
) -> bool:
  """Check if an input model's trainable layers are all registered.

  Args:
    input_model: The Keras model from which to obtain the layers from.
    layer_registry: A `LayerRegistry` instance containing functions that help
      compute gradient norms quickly. See
      `tensorflow_privacy.privacy.fast_gradient_clipping.layer_registry` for
      more details.

  Returns:
    True if all the trainable layers in `input_model` are in `layer_registry`.
    False otherwise.
  """
  for layer in input_model.layers:
    for sublayer in _get_internal_layers(layer):
      if not layer_registry.is_elem(sublayer) and sublayer.trainable_variables:
        return False
  return True


def add_aggregate_noise(
    input_model: tf.keras.Model,
    clipped_grads: list[tf.Tensor],
    batch_size: tf.Tensor,
    l2_norm_clip: float,
    noise_multiplier: float,
) -> List[tf.Tensor]:
  """Adds noise to a collection of clipped gradients.

  The magnitude of the noise depends on the aggregation strategy of the
  input model's loss function.

  Args:
    input_model: The `tf.keras.Model` to obtain the layers from.
    clipped_grads: A list of `tf.Tensor`s representing the clipped gradients.
    batch_size: The batch size, used for normalizing the noise, when the loss
      reduction is AUTO or SUM_OVER_BATCH_SIZE.
    l2_norm_clip: Clipping norm (max L2 norm of each gradient).
    noise_multiplier: Ratio of the standard deviation to the clipping norm.

  Returns:
    A list of tensors containing the clipped gradients, but with the right
    amount of Gaussian noise added to them (depending on the reduction
    strategy of the loss function).
  """
  scale = l2_norm_clip
  if input_model.loss.reduction in [
      tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
      tf.keras.losses.Reduction.AUTO,
  ]:
    if input_model.loss.reduction == tf.keras.losses.Reduction.AUTO:
      logging.info('Assuming that the loss reduction is `SUM_OVER_BATCH_SIZE`.')
    scale /= tf.cast(batch_size, tf.float32)

  def add_noise(g):
    return g + tf.random.normal(
        tf.shape(g), mean=0.0, stddev=noise_multiplier * scale
    )

  return tf.nest.map_structure(add_noise, clipped_grads)


def _make_unique_layer_names(
    layer_list: List[tf.keras.layers.Layer],
    layer_name_prefix_to_count: Dict[str, int],
):
  """Deduplicates the layer names in a layer list based on a lookup dictionary.

  Also updates the lookup dictionary with new names and counts when the input
  layer list is traversed.

  Args:
    layer_list: A `list` containing layers to deduplicate.
    layer_name_prefix_to_count: A `dict` that maps a layer name's prefix to the
      largest value of its suffix.
  """
  for l in layer_list:
    split_token = '_deduped_'
    splits = l._name.rsplit(split_token, 1)  # pylint: disable=protected-access
    prefix = splits[0]
    suffix = ''

    if len(splits) > 1:
      if splits[1].isdigit():
        prefix = splits[0] + split_token + splits[1]
      else:
        suffix = splits[1]

    if prefix not in layer_name_prefix_to_count:
      layer_name_prefix_to_count[prefix] = 0
      l._name = prefix + split_token + '0'  # pylint: disable=protected-access
    else:
      count = int(suffix) if suffix.isdigit() else 0
      if count <= layer_name_prefix_to_count[prefix]:
        layer_name_prefix_to_count[prefix] += 1
        l._name = (  # pylint: disable=protected-access
            prefix + split_token + str(layer_name_prefix_to_count[prefix])
        )


def generate_model_outputs_using_core_keras_layers(
    input_model: tf.keras.Model,
    excluded_layers: Optional[List[tf.keras.layers.Layer]] = None,
) -> PackedTensors:
  """Returns the model outputs generated by only core Keras layers.

  Args:
    input_model: A `tf.keras.Model` instance to obtain outputs from.
    excluded_layers: A `list` of layers to exclude from the expansion.

  Returns:
    A `tf.Tensor` that is the result of `input_model(input_model.inputs)`
    using only core Keras layers or excluded layers.
  """
  cust_obj_dict = dict.copy(tf.keras.utils.get_custom_objects())
  cust_hash_set = set([hash(v) for v in cust_obj_dict.values()])
  excl_hash_set = (
      set([hash(v) for v in excluded_layers])
      if excluded_layers is not None
      else set()
  )

  def is_extendable(layer_instance):
    layer_hash = hash(layer_instance.__class__)
    if excl_hash_set is None:
      return layer_hash in cust_hash_set
    else:
      return layer_hash in cust_hash_set and layer_hash not in excl_hash_set

  # Return early if all the existing layers contain only core layers.
  contains_only_core_layers = True
  for layer_instance in input_model.layers:
    if is_extendable(layer_instance):
      contains_only_core_layers = False
      break
  if contains_only_core_layers:
    return model_forward_pass(input_model, input_model.inputs)[0]

  def generator_fn(layer_instance, args, kwargs):
    # Using `.call()` does not register the layer in the compute graph of
    # a forward pass.
    layer_outputs = (
        layer_instance.call(*args, **kwargs)
        if is_extendable(layer_instance)
        else layer_instance(*args, **kwargs)
    )
    return layer_outputs, None

  # Do a forward pass to expand the outermost layers.
  candidate_outputs, _ = model_forward_pass(
      input_model, input_model.inputs, generator_fn
  )

  # De-duplicate layer names.
  layer_list = []
  tgu.depth_first_backward_pass(candidate_outputs, layer_list.append)
  _make_unique_layer_names(layer_list, {})

  # The following recursion is inefficient, but appears to be the only valid
  # approach without accessing Keras's internal functions (e.g.,
  # `keras.engine.functional._map_graph_network()`).
  cleaned_model = tf.keras.Model(
      inputs=input_model.inputs, outputs=candidate_outputs
  )
  return generate_model_outputs_using_core_keras_layers(
      cleaned_model, excluded_layers
  )
