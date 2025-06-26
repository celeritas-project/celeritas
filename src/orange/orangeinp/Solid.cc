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
namespace
{
// Overload eumod to work with turn
constexpr auto eumod(RealTurn numer, RealTurn denom)
{
    return RealTurn{celeritas::eumod(numer.value(), denom.value())};
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a starting angle and stop angle.
 */
EnclosedAzi::EnclosedAzi(Turn start, Turn stop) : start_{start}, stop_{stop}
{
    CELER_VALIDATE(stop_ > start_ && stop_ - start_ <= Turn{1},
                   << "invalid stop angle " << stop.value()
                   << " [turns]: must be in (" << start_.value() << ", "
                   << start_.value() + 1 << "]");

    if (start_ < Turn{0} || start_ >= Turn{1})
    {
        auto orig_start = start_;
        start_ = eumod(start_, Turn{1});
        stop_ = stop_ + (start_ - orig_start);
    }
    CELER_ENSURE(start_ >= Turn{0} && stop_ > start_
                 && stop_ - start_ <= Turn{1});
}

//---------------------------------------------------------------------------//
/*!
 * Construct a wedge shape to intersect (inside) or subtract (outside).
 *
 * The resulting wedge must be less than a half turn.
 */
auto EnclosedAzi::make_sense_region() const -> SenseWedge
{
    CELER_EXPECT(*this);

    if (stop_ - start_ <= Turn{0.5})
    {
        // Wedge is already an "inside" shape
        return {Sense::inside, InfAziWedge{start_, stop_}};
    }

    // Subtract the complement of the wedge: add one turn to the start value
    return {Sense::outside, InfAziWedge{stop_, start_ + Turn{1}}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId SolidBase::build(VolumeBuilder& vb) const
{
    std::vector<NodeId> nodes;

    // Build the outside-of-the-shell node
    nodes.push_back(
        build_intersect_region(vb, this->label(), "int", this->interior()));

    if (auto* exclu = this->excluded())
    {
        // Construct the excluded region by building a convex solid, then
        // negating it
        NodeId smaller
            = build_intersect_region(vb, this->label(), "exc", *exclu);
        nodes.push_back(vb.insert_region({}, Negated{smaller}));
    }

    if (auto const& azi = this->enclosed_azi())
    {
        // The user is truncating the shape azimuthally: construct a wedge to
        // be added or deleted
        auto&& [sense, wedge] = azi.make_sense_region();
        NodeId wedge_id
            = build_intersect_region(vb, this->label(), "azi", wedge);
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
 * Return a solid or shape given an optional interior or azi angle.
 */
template<class T>
auto Solid<T>::or_shape(std::string&& label,
                        T&& interior,
                        OptionalRegion&& excluded,
                        EnclosedAzi&& azi) -> SPConstObject
{
    if (!excluded && !azi)
    {
        // Just a shape
        return std::make_shared<Shape<T>>(std::move(label),
                                          std::move(interior));
    }

    return std::make_shared<Solid<T>>(std::move(label),
                                      std::move(interior),
                                      std::move(excluded),
                                      std::move(azi));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optional optional excluded region and azi angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label,
                T&& interior,
                OptionalRegion&& excluded,
                EnclosedAzi&& azi)
    : label_{std::move(label)}
    , interior_{std::move(interior)}
    , exclusion_{std::move(excluded)}
    , azi_{std::move(azi)}
{
    CELER_VALIDATE(exclusion_ || azi_,
                   << "solid requires either an excluded region or a shape");
    CELER_VALIDATE(!exclusion_ || interior_.encloses(*exclusion_),
                   << "solid '" << this->label()
                   << "' was given an interior region that is not azi by "
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
 * Construct with an azi angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label, T&& interior, EnclosedAzi&& azi)
    : Solid{std::move(label), std::move(interior), std::nullopt, std::move(azi)}
{
    CELER_VALIDATE(azi_,
                   << "solid '" << this->label()
                   << "' did not exclude an interior or a wedge (use a Shape "
                      "instead)");
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Solid<Cone>;
template class Solid<Cylinder>;
template class Solid<Prism>;
template class Solid<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
