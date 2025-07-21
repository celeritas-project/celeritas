//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Solid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/math/Turn.hh"

#include "IntersectRegion.hh"
#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Define the azimuthal truncation of a solid.
 *
 * This is a pie slice infinite along the z axis and outward from it. Its cross
 * section is in the \em x-y plane, and a start
 * angle of zero corresponding to the \em +x axis. An interior angle of one
 * results in no radial exclusion from the resulting solid. A interior angle of
 * more than 0.5 turns (180 degrees) results in a wedge being subtracted from
 * the solid, and an angle of less than or equal to 0.5 turns results in the
 * intersection of the solid with a wedge.
 *
 * \code
  // Truncates a solid to the east-facing quadrant:
  EnclosedAzi{Turn{-0.125}, Turn{0.125}};
  // Removes the second quadrant (northwest) from a solid:
  EnclosedAzi{Turn{0.50}, Turn{1.25}};
  \endcode
 */
class EnclosedAzi
{
  public:
    //!@{
    //! \name Type aliases
    using SenseWedge = std::pair<Sense, InfAziWedge>;
    //!@}

  public:
    //! Default to "all angles"
    EnclosedAzi() = default;

    // Construct from a starting angle and stop angle
    EnclosedAzi(Turn start, Turn stop);

    // Construct a wedge shape to intersect (inside) or subtract (outside)
    SenseWedge make_sense_region() const;

    // Whether the enclosed angle is not a full circle
    constexpr explicit inline operator bool() const;

    //! Starting angle
    Turn start() const { return start_; }

    //! stop angle
    Turn stop() const { return stop_; }

  private:
    Turn start_{0};
    Turn stop_{1};
};

//---------------------------------------------------------------------------//
/*!
 * Define the polar truncation of a solid.
 *
 * This subtracts up to two infinite cones centered along the z axis from the
 * origin.
 *
 * A start angle of zero corresponding to the \em +z axis. An interior angle of
 * 0.5 results in no exclusion from the resulting solid.
 * \code
  // Truncates a solid to the top hemisphere (no cones, just equatorial plane)
  EnclosedPolar{Turn{0}, Turn{0.25}};
  // Truncates a solid to northern latitudes (intersect two cones and a plane)
  EnclosedPolar{Turn{0.15}, Turn{0.2}};
  // Truncates a solid to an equatorial region (18 degrees N to 36 S: the union
  // of two polar wedge cones)
  EnclosedPolar{Turn{0.2}, Turn{0.35}};
  \endcode
 */
class EnclosedPolar
{
  public:
    //!@{
    //! \name Type aliases
    using VecPolarWedge = std::vector<InfPolarWedge>;
    //!@}

  public:
    //! Default to "all angles"
    EnclosedPolar() = default;

    // Construct from a starting angle and stop angle
    EnclosedPolar(Turn start, Turn stop);

    // Construct one or two wedges to union then intersect with the solid
    VecPolarWedge make_regions() const;

    // Whether the enclosed angle is not a full circle
    constexpr explicit inline operator bool() const;

    //! Starting angle
    Turn start() const { return start_; }

    //! stop angle
    Turn stop() const { return stop_; }

  private:
    Turn start_{0};
    Turn stop_{0.5};
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

    //! Interior intersect region interface for construction and access
    virtual IntersectRegionInterface const& interior() const = 0;

    //! Optional excluded region
    virtual IntersectRegionInterface const* excluded() const = 0;

    //! Optional azimuthal angular restriction
    virtual EnclosedAzi const& enclosed_azi() const = 0;

    //! Optional polar angular restriction
    virtual EnclosedPolar const& enclosed_polar() const = 0;

    ~SolidBase() override = default;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    SolidBase() = default;
    CELER_DEFAULT_COPY_MOVE(SolidBase);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * A shape that has undergone an intersection or combination of intersections.
 *
 * This shape may be:
 * A) hollow (excluded interior),
 * B) truncated azimuthally (enclosed angle),
 * C) truncated in z (intersected with z-slab),
 * D) both A and B.
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
    static_assert(std::is_base_of_v<IntersectRegionInterface, T>);

  public:
    //!@{
    //! \name Type aliases
    using OptionalRegion = std::optional<T>;
    //!@}

  public:
    // Return a solid *or* shape given an optional interior or enclosed angle
    static SPConstObject or_shape(std::string&& label,
                                  T&& interior,
                                  OptionalRegion&& excluded = {},
                                  EnclosedAzi&& enclosed = {},
                                  EnclosedPolar&& polar = {});

    // Construct with everything
    Solid(std::string&& label,
          T&& interior,
          OptionalRegion&& excluded,
          EnclosedAzi&& enclosed,
          EnclosedPolar&& polar);

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Interior intersect region interface for construction and access
    IntersectRegionInterface const& interior() const final
    {
        return interior_;
    }

    // Optional excluded
    IntersectRegionInterface const* excluded() const final;

    //! Optional azimuthal restriction
    EnclosedAzi const& enclosed_azi() const final { return azi_; }

    //! Optional polar restriction
    EnclosedPolar const& enclosed_polar() const final { return polar_; }

  private:
    std::string label_;
    T interior_;
    OptionalRegion exclusion_;
    EnclosedAzi azi_;
    EnclosedPolar polar_;
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
using PrismSolid = Solid<Prism>;
using SphereSolid = Solid<Sphere>;
using EllipsoidSolid = Solid<Ellipsoid>;

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class Solid<Cone>;
extern template class Solid<Cylinder>;
// TODO: hyperboloid
extern template class Solid<Prism>;
extern template class Solid<Sphere>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the enclosed angle is not a full circle.
 *
 * Note that the constructor does not allow a full circle, so only the default
 * constructor can set values of zero and 1.
 */
constexpr EnclosedAzi::operator bool() const
{
    return !(start_ == Turn{0} && stop_ == Turn{1});
}

//---------------------------------------------------------------------------//
/*!
 * Whether the enclosed angle is less than the whole polar range.
 */
constexpr EnclosedPolar::operator bool() const
{
    return !(start_ == Turn{0} && stop_ == Turn{0.5});
}

//---------------------------------------------------------------------------//
/*!
 * Access the optional excluded.
 */
template<class T>
IntersectRegionInterface const* Solid<T>::excluded() const
{
    return exclusion_ ? &(*exclusion_) : nullptr;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
