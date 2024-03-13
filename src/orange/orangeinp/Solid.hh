//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Solid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include "corecel/math/Turn.hh"

#include "ConvexRegion.hh"
#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Define the angular region of a solid.
 *
 * This is a pie slice infinite along the z axis and outward from it. Its cross
 * section is in the \em x-y plane, and a start
 * angle of zero corresponding to the \em +x axis. An interior angle of one
 * results in no radial excluded in the resulting solid. A interior angle of
 * more than 0.5 turns (180 degrees) results in a wedge being subtracted from
 * the solid, and an angle of less than or equal to 0.5 turns results in the
 * intersection of the solid with a wedge.
 *
 * \code
  // Truncates a solid to the east-facing quadrant:
  SolidEnclosedAngle{Turn{-0.125}, Turn{0.25}};
  // Removes the second quadrant (northwest) from a solid:
  SolidEnclosedAngle{Turn{0.50}, Turn{0.75}};
  \endcode
 */
class SolidEnclosedAngle
{
  public:
    //!@{
    //! \name Type aliases
    using SenseWedge = std::pair<Sense, InfWedge>;
    //!@}

  public:
    //! Default to "all angles"
    SolidEnclosedAngle() = default;

    // Construct from a starting angle and interior angle
    SolidEnclosedAngle(Turn start, Turn interior);

    // Construct a wedge shape to intersect (inside) or subtract (outside)
    SenseWedge make_wedge() const;

    // Whether the enclosed angle is not a full circle
    explicit inline operator bool() const;

    //! Starting angle
    Turn start() const { return start_; }

    //! Interior angle
    Turn interior() const { return interior_; }

  private:
    Turn start_{0};
    Turn interior_{1};
};

//---------------------------------------------------------------------------//
/*!
 * A hollow shape with an optional start and end angle.
 *
 * Solids are a shape with (optionally) the same *kind* of shape subtracted
 * from it, and (optionally) an azimuthal section removed from it.
 */
class SolidBase : public ObjectInterface
{
  public:
    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    // Write the shape to JSON
    void output(JsonPimpl*) const final;

    //! Interior convex region interface for construction and access
    virtual ConvexRegionInterface const& interior() const = 0;

    //! Optional excluded region
    virtual ConvexRegionInterface const* excluded() const = 0;

    //! Angular restriction to add
    virtual SolidEnclosedAngle enclosed_angle() const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    SolidBase() = default;
    virtual ~SolidBase() = default;
    CELER_DEFAULT_COPY_MOVE(SolidBase);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * A shape that is hollow, is truncated azimuthally, or both.
 *
 * Examples: \code
   // A cone with a thickness of 0.1
   Solid s{"cone", Cone{{1, 2}, 10.0}, Cone{{0.9, 1.9}, 10.0}};
   // A cylinder segment in z={-2.5, 2.5}, r={0.5, 0.75}, theta={-45, 45} deg
   Solid s{"cyl", Cylinder{0.75, 5.0}, Cylinder{0.5, 5.0},
           {Turn{0}, Turn{0.25}};
   // The east-facing quarter of a cone shape
   Solid s{"cone", Cone{{1, 2}, 10.0}, {Turn{-0.125}, Turn{0.25}};
 * \endcode
 */
template<class T>
class Solid final : public SolidBase
{
    static_assert(std::is_base_of_v<ConvexRegionInterface, T>);

  public:
    // Construct with an excluded interior and enclosed angle
    Solid(std::string&& label,
          T&& interior,
          T&& excluded,
          SolidEnclosedAngle enclosed);

    // Construct with an less-than-full enclosed angle
    Solid(std::string&& label, T&& interior, SolidEnclosedAngle enclosed);

    // Construct with only an excluded interior
    Solid(std::string&& label, T&& interior, T&& excluded);

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Interior convex region interface for construction and access
    ConvexRegionInterface const& interior() const final { return interior_; }

    // Optional excluded
    ConvexRegionInterface const* excluded() const final;

    //! Optional angular restriction
    SolidEnclosedAngle enclosed_angle() const final { return enclosed_; }

  private:
    std::string label_;
    T interior_;
    std::optional<T> exclusion_;
    SolidEnclosedAngle enclosed_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class T, class... Us>
Solid(std::string&&, T&&, Us...) -> Solid<T>;

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using ConeSolid = Solid<Cone>;
using CylinderSolid = Solid<Cylinder>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the enclosed angle is not a full circle.
 */
SolidEnclosedAngle::operator bool() const
{
    return start_ != Turn{0} || interior_ != Turn{1};
}

//---------------------------------------------------------------------------//
/*!
 * Access the optional excluded.
 */
template<class T>
ConvexRegionInterface const* Solid<T>::excluded() const
{
    return exclusion_ ? &(*exclusion_) : nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
