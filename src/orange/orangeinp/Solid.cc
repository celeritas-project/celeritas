//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Solid.cc
//---------------------------------------------------------------------------//
#include "Solid.hh"

#include "corecel/io/JsonPimpl.hh"
#include "corecel/math/Algorithms.hh"

#include "ConvexSurfaceBuilder.hh"
#include "CsgTreeUtils.hh"

#include "detail/BoundingZone.hh"
#include "detail/BuildConvexRegion.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/VolumeBuilder.hh"

#if CELERITAS_USE_JSON
#    include "ObjectIO.json.hh"
#endif

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a starting angle and interior angle.
 */
SolidEnclosedAngle::SolidEnclosedAngle(Turn start, Turn interior)
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
auto SolidEnclosedAngle::make_wedge() const -> SenseWedge
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
 * Construct a volume from this shape.
 */
NodeId SolidBase::build(VolumeBuilder& vb) const
{
    std::vector<NodeId> nodes;

    // Build the outside-of-the-shell node
    nodes.push_back(build_convex_region(
        vb, std::string{this->label()}, "interior", this->interior()));

    if (auto* exclu = this->excluded())
    {
        // Construct the excluded region by building a convex solid, then
        // negating it
        NodeId smaller = build_convex_region(
            vb, std::string{this->label()}, "excluded", *exclu);
        nodes.push_back(vb.insert_region({}, Negated{smaller}));
    }

    if (auto const& sea = this->enclosed_angle())
    {
        // The enclosed angle is "true" (specified by the user to truncate the
        // shape azimuthally): construct a wedge to be added or deleted
        auto&& [sense, wedge] = sea.make_wedge();
        NodeId wedge_id = build_convex_region(
            vb, std::string{this->label()}, "angle", wedge);
        if (sense == Sense::outside)
        {
            wedge_id = vb.insert_region({}, Negated{wedge_id});
        }
        nodes.push_back(wedge_id);
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
 * Construct with an excluded interior and enclosed angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label,
                T&& interior,
                T&& excluded,
                SolidEnclosedAngle enclosed)
    : label_{std::move(label)}
    , interior_{std::move(interior)}
    , exclusion_{std::move(excluded)}
    , enclosed_(enclosed)
{
    CELER_VALIDATE(interior_.encloses(*exclusion_),
                   << "solid '" << this->label()
                   << "' was given an interior region that is not enclosed by "
                      "its exterior");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with an enclosed angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label, T&& interior, SolidEnclosedAngle enclosed)
    : label_{std::move(label)}
    , interior_{std::move(interior)}
    , enclosed_(enclosed)
{
    CELER_VALIDATE(enclosed_,
                   << "solid '" << this->label()
                   << "' did not exclude an interior or a wedge (use a Shape "
                      "instead)");
}

//---------------------------------------------------------------------------//
/*!
 * Construct with only an excluded interior.
 */
template<class T>
Solid<T>::Solid(std::string&& label, T&& interior, T&& excluded)
    : Solid{std::move(label),
            std::move(interior),
            std::move(excluded),
            SolidEnclosedAngle{}}
{
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Solid<Cone>;
template class Solid<Cylinder>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
