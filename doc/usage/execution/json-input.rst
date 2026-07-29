.. Copyright Celeritas contributors: see top-level COPYRIGHT file for details
.. SPDX-License-Identifier: CC-BY-4.0

.. _json-input:

JSON input for standalone apps
==============================

Celeritas standalone applications such as ``celer-sim`` and ``celer-optical``
accept structured JSON input based on the ``celeritas::inp`` data structures.
The :ref:`input` documents the underlying C++ types; this page explains how
those types are represented in JSON and gives minimal working examples.

JSON encoding conventions
-------------------------

Structs
^^^^^^^

A C++ struct is represented as a JSON object with matching field names.

For example, a struct like:

.. code-block:: cpp

   struct Timers
   {
       bool action{false};
       bool step{false};
   };

becomes a JSON object where the keys have the same names as the members:

.. code-block:: json

   {
     "action": false,
     "step": false
   }

Arrays, maps, and sets
^^^^^^^^^^^^^^^^^^^^^^

``std::vector<T>`` and ``celeritas::Array<T>`` values become JSON arrays.
``std::unordered_set<T>`` values become JSON arrays of unique values, and
``std::map<K, V>`` values become JSON objects.

Enums
^^^^^

Celeritas enumerations are encoded in JSON as strings rather than integer
values.

For example, the C++ enumeration

.. code-block:: cpp

   enum class InterpolationType
   {
       linear,
       poly_spline,
       cubic_spline,
       size_
   };

is represented in JSON using the corresponding string values: ``"linear"``,
``"poly_spline"``, and ``"cubic_spline"``. A field ``InterpolationType type``
whose value is ``InterpolationType::linear`` in JSON becomes:

.. code-block:: json

   {
     "type": "linear"
   }

Optional values
^^^^^^^^^^^^^^^

Fields with type ``std::optional<T>`` support several different JSON behaviors:

* If the field is omitted, the existing value is unchanged. This allows the
  containing input structure to define a custom default value.
* If the field is set to ``null``, the optional is reset to
  ``std::nullopt``.
* If the field is set to a non-null value, the optional is initialized to the
  given value.
* If the field is set to ``{}``, the optional is initialized with a
  default-constructed ``T``.

For example, omitting the optional ``device`` field in the :ref:`inp_system`
input preserves the existing value supplied by the containing structure (i.e.,
the device remains disabled):

.. code-block:: json

   {
     "system": {}
   }

Setting the field to ``null`` explicitly clears the optional value (i.e.,
disables the device):

.. code-block:: json

   {
     "system": {
       "device": null
     }
   }

Setting the field to an object initializes the optional and reads the nested
value (i.e., enables the device with custom stack and heap sizes):

.. code-block:: json

   {
     "system": {
       "device": {
         "stack_size": 1024,
         "heap_size": 8192
       }
     }
   }

Setting the field to an empty object initializes the optional with the
contained type's default (i.e., enables the device with the default stack and
heap size):

.. code-block:: json

   {
     "system": {
       "device": {}
     }
   }

This means omission and ``{}`` are not equivalent: omission preserves the value
that was already present, while ``{}`` replaces it with an initialized contained
object using default values.

Variants
^^^^^^^^

Many Celeritas input types are ``std::variant`` objects. In JSON, these are
encoded as objects with a required ``_type`` field that identifies the selected
alternative.

For example, the ``Field`` input type is defined as:

.. code-block:: cpp

   using Field = std::variant<NoField,
                              UniformField,
                              RZMapField,
                              CylMapField,
                              CartMapField>;

A C++ value such as

.. code-block:: cpp

   inp::Field field = inp::NoField{};

is represented in JSON as:

.. code-block:: json

   {
     "_type": "none"
   }

Similarly, a uniform magnetic field can be specified in C++ as:

.. code-block:: cpp

   inp::Field field = [] {
       inp::UniformField result;
       result.strength = {0.0, 0.0, 1.0};
       return result;
   }();

and in JSON as:

.. code-block:: json

   {
     "_type": "uniform",
     "strength": [0.0, 0.0, 1.0]
   }

The ``_type`` field selects the variant alternative, and the remaining fields
specify the data for that alternative. The ``_type`` value does not necessarily
match the C++ type name. Refer to the documented string values for the variant
being encoded.

See :ref:`json-events-example` for an example combining several variant-valued
fields.

.. _json-events-example:

Example: encoding ``inp::Events`` in JSON
-----------------------------------------

The ``inp::Events`` input type is a useful example of Celeritas JSON encoding
because it combines several common patterns in one place.

A C++ setup constructing a primary generator for ``celer-sim`` might look like:

.. code-block:: cpp

   inp::Events events = [] {
       inp::CorePrimaryGenerator result;
       result.shape = inp::PointDistribution{{0.0, 0.0, 0.0}};
       result.angle = inp::IsotropicDistribution{};
       result.energy = inp::MonoenergeticDistribution{10.0};
       result.num_events = 1;
       result.primaries_per_event = 10000;
       result.pdg = {PDGNumber{11}};
   }();
   events.merge = false;

In JSON, this is represented as:

.. code-block:: json

   {
     "generator": {
       "_type": "primary",
       "shape": {
         "_type": "delta",
         "value": [0.0, 0.0, 0.0]
       },
       "angle": {
         "_type": "isotropic"
       },
       "energy": {
         "_type": "delta",
         "value": 10.0
       },
       "num_events": 1,
       "primaries_per_event": 10000,
       "pdg": [11]
     },
     "merge": false
   }

This example illustrates several of the JSON encoding conventions used by
Celeritas:

* ``generator`` is a ``std::variant`` and is encoded with a ``_type`` field.
* ``shape``, ``angle``, and ``energy`` are also variants and each uses its own
  ``_type`` field.
* The ``pdg`` vector is encoded as a JSON array.
* Scalar fields such as ``num_events`` and ``merge`` are encoded directly as
  JSON numbers and booleans.

See also
--------

* :ref:`input`
* :ref:`celer-sim`
* :ref:`celer-optical`
