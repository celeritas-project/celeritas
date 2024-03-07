//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Shape.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "ConvexRegion.hh"
#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * A simple, convex region of space.
 *
 * This is an abstract class that implements \c build for constructing a volume
 * by dispatching to a method \c build_interior that the daughters must
 * override using a convex region.
 *
 * Use the implementation classes \c XShape where \c X is one of the convex
 * region types in ConvexRegion.hh :
 * - \c BoxShape
 * - \c ConeShape
 * - \c CylinderShape
 * - \c EllipsoidShape
 * - \c PrismShape
 * - \c SphereShape
 */
class ShapeBase : public ObjectInterface
{
  public:
    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

    // Write the shape to JSON
    void output(JsonPimpl*) const final;

    //! Interior convex region interface for construction and access
    virtual ConvexRegionInterface const& interior() const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ShapeBase() = default;
    virtual ~ShapeBase() = default;
    CELER_DEFAULT_COPY_MOVE(ShapeBase);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Shape that holds a convex region and forwards construction args to it.
 *
 * Construct as: \code
    BoxShape s{"mybox", Real3{1, 2, 3}};
 * \endcode
 * or \code
 *  Shape s{"mybox", Box{{1, 2, 3}}};
 * \endcode
 *
 * See ConvexRegion.hh for a list of the regions and their construction
 * arguments.
 */
template<class T>
class Shape final : public ShapeBase
{
    static_assert(std::is_base_of_v<ConvexRegionInterface, T>);

  public:
    //! Construct with a label and arguments of the convex region
    template<class... Ts>
    Shape(std::string&& label, Ts... region_args)
        : label_{std::move(label)}, region_{std::forward<Ts>(region_args)...}
    {
        CELER_EXPECT(!label_.empty());
    }

    //! Construct with a label and convex region
    Shape(std::string&& label, T&& region)
        : label_{std::move(label)}, region_{std::move(region)}
    {
        CELER_EXPECT(!label_.empty());
    }

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Interior convex region
    ConvexRegionInterface const& interior() const final { return region_; }

  private:
    std::string label_;
    T region_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class T>
Shape(std::string&&, T&&) -> Shape<T>;

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using BoxShape = Shape<Box>;
using ConeShape = Shape<Cone>;
using CylinderShape = Shape<Cylinder>;
using EllipsoidShape = Shape<Ellipsoid>;
using PrismShape = Shape<Prism>;
using SphereShape = Shape<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
