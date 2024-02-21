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
 * Use the implementation classes \c XShape where \c X is one of the convex
 * region types in ConvexRegion.hh :
 * - \c BoxShape
 * - \c ConeShape
 * - \c CylinderShape
 * - \c EllipsoidShape
 * - \c PrismShape
 * - \c SphereShape
 */
class Shape : public ObjectInterface
{
  public:
    // Construct a volume from this object
    NodeId build(VolumeBuilder&) const final;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    Shape() = default;
    virtual ~Shape() = default;
    CELER_DEFAULT_COPY_MOVE(Shape);
    //!@}

    //! Daughter class interface
    virtual void build_interior(ConvexSurfaceBuilder&) const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Shape that holds a convex region and forwards construction args to it.
 *
 * Construct as:
 *
 *    BoxShape s{"mybox", Real3{1, 2, 3}};
 *
 * See ConvexRegion.hh for a list of the regions and their construction
 * arguments.
 */
template<class T>
class ShapeImpl final : public Shape
{
    static_assert(std::is_base_of_v<ConvexRegionInterface, T>);

  public:
    //! Construct with a label and arguments of the convex region
    template<class... Ts>
    ShapeImpl(std::string&& label, Ts... region_args)
        : label_{std::move(label)}, region_{std::forward<Ts>(region_args)...}
    {
        CELER_EXPECT(!label_.empty());
    }

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Forward the construction call to the convex region
    void build_interior(ConvexSurfaceBuilder& csb) const final
    {
        return region_.build(csb);
    }

  private:
    std::string label_;
    T region_;
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

using BoxShape = ShapeImpl<Box>;
using ConeShape = ShapeImpl<Cone>;
using CylinderShape = ShapeImpl<Cylinder>;
using EllipsoidShape = ShapeImpl<Ellipsoid>;
using PrismShape = ShapeImpl<Prism>;
using SphereShape = ShapeImpl<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
