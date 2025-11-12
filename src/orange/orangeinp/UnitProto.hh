//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "geocel/Types.hh"
#include "orange/OrangeTypes.hh"
#include "orange/transform/VariantTransform.hh"

#include "CsgTypes.hh"
#include "ProtoInterface.hh"

namespace celeritas
{
namespace orangeinp
{
class Transformed;

namespace detail
{
struct CsgUnit;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Construct a general CSG universe, aka a "unit".
 *
 * A "unit" is a region of space (with an outer boundary specified by the \c
 * BoundaryInput::interior object) that is divided up into multiple smaller
 * regions:
 * - A "material" (aka \em media in SCALE) is a single homogeneous CSG object
 *   filled with a particular material ID. This is equivalent to a leaf
 *   "physical volume" in a GDML/Geant4 volume hierarchy.
 * - A "daughter" (aka \em hole in SCALE) is another unit that is transformed
 *   and placed into this universe.
 *
 * Additional metadata about the structure can be provided when converting
 * Geant4 geometry. When building, labels are volume instance IDs: each
 * material and daughter is a volume placement, and the background input uses
 * an empty instance ID as a sentinel indicating a background volume is
 * present. (Note the "background" is structurally a \em volume, not \em
 * instance, since it is locally the top level of the geometry.)
 * In addition to the labels, a special \c local_parent field denotes a
 * structural relationship between one input object and another. These will be
 * always set when building from Geant4 but never set when building from SCALE.
 * The value is a null ID when the parent canonical volume is rendered as the
 * background, or the index in the list of \c materials when the parent volume
 * is \em inlined into this unit.
 */
class UnitProto : public ProtoInterface
{
  public:
    //!@{
    //! \name Types
    using Unit = detail::CsgUnit;
    using Tol = Tolerance<>;
    using VariantLabel = std::variant<Label, VolumeInstanceId>;
    using MaterialInputId = OpaqueId<struct MaterialInput>;
    using LocalParent = std::optional<MaterialInputId>;
    //!@}

    //! Optional "background" inside of exterior, outside of all mat/daughter
    struct BackgroundInput
    {
        GeoMatId fill{};
        VariantLabel label;

        // True if fill or label is specified
        explicit inline operator bool() const;
    };

    //! A homogeneous physical material
    struct MaterialInput
    {
        SPConstObject interior;
        GeoMatId fill;
        VariantLabel label;
        //! Mark this material as being structurally inside another local one
        LocalParent local_parent;

        // True if fully defined
        explicit inline operator bool() const;
    };

    //! Another universe embedded within this one
    struct DaughterInput
    {
        SPConstProto fill;  //!< Daughter unit
        VariantTransform transform;  //!< Daughter-to-parent
        ZOrder zorder{ZOrder::media};  //!< Overlap control
        VariantLabel label;  //!< Placement name

        //! Mark this daughter as being inside another local volume
        LocalParent local_parent;

        // True if fully defined
        explicit inline operator bool() const;

        // Construct the daughter's shape in this unit's reference frame
        SPConstObject make_interior() const;
    };

    //! Boundary conditions for the unit
    struct BoundaryInput
    {
        SPConstObject interior;  //!< Bounding shape for the unit
        ZOrder zorder{ZOrder::exterior};  //!< Overlap control

        // True if fully defined
        explicit inline operator bool() const;
    };

    //! Required input data to create a unit proto
    struct Input
    {
        BackgroundInput background;
        std::vector<MaterialInput> materials;
        std::vector<DaughterInput> daughters;
        BoundaryInput boundary;
        Label label;

        //!@{
        //! \name Construction options

        //! For non-global units, assume inside the boundary
        bool remove_interior{true};
        //! Use DeMorgan's law to remove negated joins
        bool remove_negated_join{false};

        //!@}

        // True if fully defined
        explicit inline operator bool() const;
    };

  public:
    // Construct with required input data
    explicit UnitProto(Input&& inp);

    virtual ~UnitProto() = default;

    // Short unique name of this object
    std::string_view label() const final;

    // Get the boundary of this universe as an object
    SPConstObject interior() const final;

    // Get a list of all daughters
    VecProto daughters() const final;

    // Construct a universe input from this object
    void build(ProtoBuilder&) const final;

    // Write the proto to a JSON object
    void output(JsonPimpl*) const final;

    //// ACCESSORS ////

    //! Get the input, primarily for unit testing
    Input const& input() const { return input_; }

    //// HELPER FUNCTIONS ////

    // Construct a standalone unit for testing and external interface
    Unit build(Tol const& tol, BBox const& bbox) const;

  private:
    Input input_;
};

//---------------------------------------------------------------------------//
/*!
 * True if either material or label is provided.
 */
UnitProto::BackgroundInput::operator bool() const
{
    return static_cast<bool>(this->fill);
}

//---------------------------------------------------------------------------//
/*!
 * True if fully defined.
 */
UnitProto::MaterialInput::operator bool() const
{
    return static_cast<bool>(this->interior);
}

//---------------------------------------------------------------------------//
/*!
 * True if fully defined.
 */
UnitProto::DaughterInput::operator bool() const
{
    return static_cast<bool>(this->fill);
}

//---------------------------------------------------------------------------//
/*!
 * True if fully defined.
 */
UnitProto::BoundaryInput::operator bool() const
{
    return static_cast<bool>(this->interior);
}

//---------------------------------------------------------------------------//
/*!
 * True if fully defined.
 *
 * The unit proto must have at least one material, daughter, or background
 * fill.
 */
UnitProto::Input::operator bool() const
{
    return (!this->materials.empty() || !this->daughters.empty()
            || this->background)
           && this->boundary;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
