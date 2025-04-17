//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/UnitProto.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

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
 * A "unit" is a region of space (specified by the "boundary" object) that is
 * divided up into multiple smaller regions:
 * - A "material" (aka "media" in SCALE) is a single homogeneous CSG object
 *   filled with a particular material ID. This is equivalent to a leaf
 *   "physical volume" in a GDML/Geant4 volume hierarchy.
 * - A "daughter" (aka "hole" in SCALE) is another unit that is transformed and
 *   placed into this universe.
 *
 * Regarding boundary conditions: "Input" is for how the unit is *defined*:
 *
 *  ========== ==========================================================
 *   Input      Description
 *  ========== ==========================================================
 *   Implicit   Boundary implicitly truncates interior (KENO)
 *   Explicit   Interior CSG definition includes boundary (RTK)
 *  ========== ==========================================================
 *
 * Additionally, whether the universe is the top-level \em global universe (see
 * the \c ExteriorBoundary type) affects the construction.
 *
 *  ========== ==========================================================
 *   ExtBound   Description
 *  ========== ==========================================================
 *   Daughter   Boundary is already truncated by higher-level unit
 *   Global     Boundary must explicitly be represented as a volume
 *  ========== ==========================================================
 *
 * These result in different z ordering for the exterior:
 *
 *  ===== ===== ================== ========================================
 *   Inp   ExB   Resulting zorder   Description
 *  ===== ===== ================== ========================================
 *   I     N     implicit_exterior  Higher-level universe truncates
 *   X     N     implicit_exterior  Higher-level universe truncates
 *   I     Y     exterior           Global unit that truncates other regions
 *   X     Y     media              Global unit with well-connected exterior
 *  ===== ===== ================== ========================================
 */
class UnitProto : public ProtoInterface
{
  public:
    //!@{
    //! \name Types
    using Unit = detail::CsgUnit;
    using Tol = Tolerance<>;

    //! Optional "background" inside of exterior, outside of all mat/daughter
    struct BackgroundInput
    {
        GeoMatId fill{};
        Label label;

        // True if fill or label is specified
        explicit inline operator bool() const;
    };

    //! A homogeneous physical material
    struct MaterialInput
    {
        SPConstObject interior;
        GeoMatId fill;
        Label label;

        // True if fully defined
        explicit inline operator bool() const;
    };

    //! Another universe embedded within this one
    struct DaughterInput
    {
        SPConstProto fill;  //!< Daughter unit
        VariantTransform transform;  //!< Daughter-to-parent
        ZOrder zorder{ZOrder::media};  //!< Overlap control

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
        std::string label;
        UnitSimplification simplification{UnitSimplification::none};

        // True if fully defined
        explicit inline operator bool() const;
    };
    //!@}

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
    return this->fill || !this->label.empty();
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
