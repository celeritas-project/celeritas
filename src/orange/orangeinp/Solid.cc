//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Solid.cc
//---------------------------------------------------------------------------//
#include "Solid.hh"

#include "corecel/io/JsonPimpl.hh"
#include "corecel/math/Algorithms.hh"

#include "CsgTreeUtils.hh"
#include "IntersectSurfaceBuilder.hh"
#include "ObjectIO.json.hh"
#include "Shape.hh"

#include "detail/BoundingZone.hh"
#include "detail/BuildIntersectRegion.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a starting angle and interior angle.
 */
EnclosedAzi::EnclosedAzi(Turn start, Turn interior)
    : start_{start}, interior_{interior}
{
    CELER_VALIDATE(interior_ > zero_quantity() && interior_ <= Turn{1},
                   << "invalid interior angle " << interior.value()
                   << " [turns]: must be in (0, 1]");
}

//---------------------------------------------------------------------------//
/*!
 * Construct a wedge shape to intersect (inside) or subtract (outside).
 */
auto EnclosedAzi::make_sense_region() const -> SenseWedge
{
    CELER_EXPECT(*this);
    // Get the start value between [0, 1)
    real_type start = eumod(start_.value(), real_type{1});
    real_type interior = interior_.value();
    Sense sense = Sense::inside;
    if (interior > real_type{0.5})
    {
        // Subtract the complement of the wedge
        sense = Sense::outside;
        start = eumod(start + interior, real_type{1});
        interior = 1 - interior;
    }

    return {sense, InfWedge{Turn{start}, Turn{interior}}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from lower and upper z-planes.
 */
SolidZSlab::SolidZSlab(real_type lower, real_type upper)
    : lower_{lower}, upper_{upper}
{
    CELER_VALIDATE(lower < upper,
                   << "invalid lower z-plane " << lower
                   << " must be less than upper z plane " << upper);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an InfSlab shape to intersect with the solid.
 */
auto SolidZSlab::make_inf_slab() const -> InfSlab
{
    CELER_EXPECT(*this);
    return InfSlab{lower_, upper_};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId SolidBase::build(VolumeBuilder& vb) const
{
    std::vector<NodeId> nodes;

    // Build the outside-of-the-shell node
    nodes.push_back(build_intersect_region(
        vb, this->label(), "interior", this->interior()));

    if (auto* exclu = this->excluded())
    {
        // Construct the excluded region by building a convex solid, then
        // negating it
        NodeId smaller
            = build_intersect_region(vb, this->label(), "excluded", *exclu);
        nodes.push_back(vb.insert_region({}, Negated{smaller}));
    }

    if (auto const& azi = this->enclosed_azi())
    {
        // The enclosed angle is "true" (specified by the user to truncate the
        // shape azimuthally): construct a wedge to be added or deleted
        auto&& [sense, wedge] = azi.make_sense_region();
        NodeId wedge_id
            = build_intersect_region(vb, this->label(), "angle", wedge);
        if (sense == Sense::outside)
        {
            wedge_id = vb.insert_region({}, Negated{wedge_id});
        }
        nodes.push_back(wedge_id);
    }

    if (auto const& szs = this->z_slab())
    {
        nodes.push_back(build_intersect_region(
            vb, this->label(), "z_slab", szs.make_inf_slab()));
    }

    // Intersect the given surfaces to create a new CSG node
    return vb.insert_region(Label{std::string{this->label()}},
                            Joined{op_and, std::move(nodes)});
}

//---------------------------------------------------------------------------//
/*!
 * Output to JSON.
 */
void SolidBase::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
/*!
 * Return a solid or shape given an optional interior or enclosed angle.
 */
template<class T>
auto Solid<T>::or_shape(std::string&& label,
                        T&& interior,
                        OptionalRegion&& excluded,
                        EnclosedAzi&& enclosed) -> SPConstObject
{
    if (!excluded && !enclosed)
    {
        // Just a shape
        return std::make_shared<Shape<T>>(std::move(label),
                                          std::move(interior));
    }

    return std::make_shared<Solid<T>>(std::move(label),
                                      std::move(interior),
                                      std::move(excluded),
                                      std::move(enclosed));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optional optional excluded region and enclosed angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label,
                T&& interior,
                OptionalRegion&& excluded,
                EnclosedAzi&& enclosed)
    : label_{std::move(label)}
    , interior_{std::move(interior)}
    , exclusion_{std::move(excluded)}
    , enclosed_{std::move(enclosed)}
    , z_slab_{}
{
    CELER_VALIDATE(exclusion_ || enclosed_,
                   << "solid requires either an excluded region or a shape");
    CELER_VALIDATE(!exclusion_ || interior_.encloses(*exclusion_),
                   << "solid '" << this->label()
                   << "' was given an interior region that is not enclosed by "
                      "its exterior");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with an excluded interior.
 */
template<class T>
Solid<T>::Solid(std::string&& label, T&& interior, T&& excluded)
    : Solid{std::move(label),
            std::move(interior),
            std::move(excluded),
            EnclosedAzi{}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with an enclosed angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label, T&& interior, EnclosedAzi&& enclosed)
    : Solid{std::move(label),
            std::move(interior),
            std::nullopt,
            std::move(enclosed)}
{
    CELER_VALIDATE(enclosed_,
                   << "solid '" << this->label()
                   << "' did not exclude an interior or a wedge (use a Shape "
                      "instead)");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with only a z-slab.
 */
template<class T>
Solid<T>::Solid(std::string&& label, T&& interior, SolidZSlab&& z_slab)
    : label_{std::move(label)}
    , interior_{std::move(interior)}
    , exclusion_{std::nullopt}
    , enclosed_{EnclosedAzi{}}
    , z_slab_{std::move(z_slab)}
{
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Solid<Cone>;
template class Solid<Cylinder>;
template class Solid<Prism>;
template class Solid<Sphere>;
template class Solid<Ellipsoid>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
