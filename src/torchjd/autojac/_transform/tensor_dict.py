from torch import Tensor


class TensorDict(dict[Tensor, Tensor]):
    """
    Class used to represent an immutable mapping from :class:`~torch.Tensor` to
    :class:`~torch.Tensor`. It is responsible to check some properties on the mapping.

    :param tensor_dict: The mapping to check and store.
    """

    def __init__(self, tensor_dict: dict[Tensor, Tensor]):
        self._check_dict(tensor_dict)
        self._check_all_pairs(tensor_dict)
        super().__init__(tensor_dict)

    def check_keys_are(self, keys: set[Tensor]) -> None:
        """
        Checks that the keys in the mapping are the same as the provided ``keys``.

        :param keys: Keys that the mapping should (exclusively) contain.
        """

        if set(keys) != set(self.keys()):
            raise ValueError(
                f"The keys of the {self.__class__.__name__} should be {keys}. Found self.keys = "
                f"{self.keys()}."
            )

    @staticmethod
    def _check_dict(tensor_dict: dict[Tensor, Tensor]) -> None:
        pass

    @classmethod
    def _check_all_pairs(cls, tensor_dict: dict[Tensor, Tensor]) -> None:
        for key, value in tensor_dict.items():
            cls._check_key_value_pair(key, value)

    @staticmethod
    def _check_key_value_pair(key: Tensor, value: Tensor) -> None:
        pass

    # Make TensorDict immutable, following answer in
    # https://stackoverflow.com/questions/11014262/how-to-create-an-immutable-dictionary-in-python
    # coming from https://peps.python.org/pep-0351/
    def _raise_immutable_error(self, *args, **kwargs) -> None:
        raise TypeError(f"{self.__class__.__name__} is immutable.")

    __setitem__ = _raise_immutable_error
    __delitem__ = _raise_immutable_error
    clear = _raise_immutable_error
    update = _raise_immutable_error
    setdefault = _raise_immutable_error
    pop = _raise_immutable_error
    popitem = _raise_immutable_error


class Gradients(TensorDict):
    """
    :class:`~torchjd.transform.tensor_dict.TensorDict` in which the values are gradients with
    respect to the keys.

    - The shape of each value must be the same as the shape of its corresponding key.
    """

    @staticmethod
    def _check_key_value_pair(key: Tensor, value: Tensor) -> None:
        _check_same_shape(key, value)


class Jacobians(TensorDict):
    """
    :class:`~torchjd.transform.tensor_dict.TensorDict` in which the values are jacobians with
    respect to the keys.

    - The values must all have the same first dimension.
    - The rest of the shape of each value must be the same as the shape of its corresponding key.
    """

    @staticmethod
    def _check_dict(tensor_dict: dict[Tensor, Tensor]) -> None:
        _check_values_have_unique_first_dim(tensor_dict)

    @staticmethod
    def _check_key_value_pair(key: Tensor, value: Tensor) -> None:
        _check_value_has_jacobian_shape(key, value)

    @property
    def first_dimension(self) -> int:
        return _get_first_dimension(self)


class GradientVectors(TensorDict):
    """
    :class:`~torchjd.transform.tensor_dict.TensorDict` in which the values are flattened gradients
    (gradient vectors) with respect to the keys.

    - The values must be vectors with the same number of elements as their corresponding key.
    """

    @staticmethod
    def _check_key_value_pair(key: Tensor, value: Tensor) -> None:
        _check_value_n_dim(value, expected_n_dim=1)
        _check_corresponding_numel(key, value, dim=0)


class JacobianMatrices(TensorDict):
    """
    :class:`~torchjd.transform.tensor_dict.TensorDict` in which the values are matrixified jacobians
    (jacobian matrices) with respect to the keys.

    - The values must be matrices with a unique first dimension and with a second dimension equal to
      the number of elements of their corresponding key.
    """

    @staticmethod
    def _check_dict(tensor_dict: dict[Tensor, Tensor]) -> None:
        _check_values_have_unique_first_dim(tensor_dict)

    @staticmethod
    def _check_key_value_pair(key: Tensor, value: Tensor) -> None:
        _check_value_n_dim(value, expected_n_dim=2)
        _check_corresponding_numel(key, value, dim=1)

    @property
    def first_dimension(self) -> int:
        return _get_first_dimension(self)


class EmptyTensorDict(
    Gradients,
    Jacobians,
    GradientVectors,
    JacobianMatrices,
):
    """
    :class:`~torchjd.transform.tensor_dict.TensorDict` containing no element. It satisfies the
    properties of all other :class:`~torchjd.transform.tensor_dict.TensorDict` subclasses without
    explicitly checking them.
    """

    def __init__(self, tensor_dict: dict[Tensor, Tensor] | None = None):
        if tensor_dict is not None and len(tensor_dict) != 0:
            raise ValueError("Cannot build a non-empty `EmptyTensorDict`")
        super().__init__({})


def _least_common_ancestor(first: type[TensorDict], second: type[TensorDict]) -> type[TensorDict]:
    first_mro = first.mro()[:-1]  # removes `object` from `mro`.
    output = TensorDict
    for candidate_type in first_mro:
        if issubclass(second, candidate_type):
            output = candidate_type
            break
    return output


def _get_first_dimension(tensor_dict: dict[Tensor, Tensor]) -> int:
    if len([tensor_dict.keys()]) == 0:
        first_dimension = 0  # By convention, a dict without any keys has a first dimension of 0
    else:
        value = next(iter(tensor_dict.values()))
        first_dimension = value.shape[0]

    return first_dimension


def _check_values_have_unique_first_dim(tensor_dict: dict[Tensor, Tensor]) -> None:
    first_dims = [value.shape[0] for value in tensor_dict.values()]
    if len(set(first_dims)) > 1:
        raise ValueError(
            "Parameter `tensor_dict` should contain value tensors with the same first dimension. "
            f"Found the following first dimensions: `{first_dims}`."
        )


def _check_value_n_dim(value: Tensor, expected_n_dim: int) -> None:
    n_dim = value.dim()
    if n_dim != expected_n_dim:
        raise ValueError(
            f"Parameter `tensor_dict` should contain value tensors of dimension {expected_n_dim}. "
            f"Found one with dimension {n_dim}."
        )


def _check_value_has_jacobian_shape(key: Tensor, value: Tensor) -> None:
    if value.shape[1:] != key.shape:
        raise ValueError(
            "Parameter `tensor_dict` should contain value tensors of one more dimension than the "
            "corresponding key tensor, with the rest of the shape matching the shape of the key "
            f"tensor. Found pair with key shape {key.shape} and value shape {value.shape}."
        )


def _check_same_shape(key: Tensor, value: Tensor) -> None:
    if value.shape != key.shape:
        raise ValueError(
            "Parameter `tensor_dict` should contain value tensors of the same shape as the "
            f"corresponding key tensor. Found pair with key shape {key.shape} and value shape "
            f"{value.shape}."
        )


def _check_corresponding_numel(key: Tensor, value: Tensor, dim: int) -> None:
    if value.shape[dim] != key.shape.numel():
        raise ValueError(
            f"Parameter `tensor_dict` should contain value tensors with dimension {dim} equal to "
            "the number of elements in the corresponding key tensor. Found pair with key shape "
            f"{key.shape} and value shape {value.shape}."
        )
