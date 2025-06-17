//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Shape.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "IntersectRegion.hh"
#include "ObjectInterface.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * A simple, intersect-only region of space.
 *
 * This is an abstract class that implements \c build for constructing a volume
 * by dispatching to a method \c build_interior that the daughters must
 * override using a intersect region.
 *
 * Use the implementation classes \c XShape where \c X is one of the
 * region types in IntersectRegion.hh :
 * - \c BoxShape
 * - \c ConeShape
 * - \c CylinderShape
 * - \c EllipsoidShape
 * - \c GenPrismShape
 * - \c ParallelepipedShape
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

    //! Interior intersect region interface for construction and access
    virtual IntersectRegionInterface const& interior() const = 0;

    ~ShapeBase() override = default;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ShapeBase() = default;
    CELER_DEFAULT_COPY_MOVE(ShapeBase);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Shape that holds an intersect region and forwards construction args to it.
 *
 * Construct as: \code
    BoxShape s{"mybox", Real3{1, 2, 3}};
 * \endcode
 * or \code
 *  Shape s{"mybox", Box{{1, 2, 3}}};
 * \endcode
 *
 * See IntersectRegion.hh for a list of the regions and their construction
 * arguments.
 */
template<class T>
class Shape final : public ShapeBase
{
    static_assert(std::is_base_of_v<IntersectRegionInterface, T>);

  public:
    //! Construct with a label and arguments of the intersect region
    template<class... Ts>
    Shape(std::string&& label, Ts... region_args)
        : label_{std::move(label)}, region_{std::forward<Ts>(region_args)...}
    {
        CELER_EXPECT(!label_.empty());
    }

    //! Construct with a label and intersect region
    Shape(std::string&& label, T&& region)
        : label_{std::move(label)}, region_{std::move(region)}
    {
        CELER_EXPECT(!label_.empty());
    }

    //! Get the user-provided label
    std::string_view label() const final { return label_; }

    //! Interior intersect region
    IntersectRegionInterface const& interior() const final { return region_; }

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
using ExtrudedPolygonShape = Shape<ExtrudedPolygon>;
using GenPrismShape = Shape<GenPrism>;
using InvoluteShape = Shape<Involute>;
using ParallelepipedShape = Shape<Parallelepiped>;
using PrismShape = Shape<Prism>;
using SphereShape = Shape<Sphere>;

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class Shape<Box>;
extern template class Shape<Cone>;
extern template class Shape<Cylinder>;
extern template class Shape<Ellipsoid>;
extern template class Shape<ExtrudedPolygon>;
extern template class Shape<GenPrism>;
extern template class Shape<Involute>;
extern template class Shape<Parallelepiped>;
extern template class Shape<Prism>;
extern template class Shape<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
