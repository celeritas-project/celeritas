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
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
// Overload eumod to work with turn
constexpr auto eumod(RealTurn num, RealTurn denom)
{
    return RealTurn{celeritas::eumod(num.value(), denom.value())};
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
 * Construct from a starting angle and stop angle.
 *
 * The beginning starts at the north pole/top point and the end is at the south
 * pole/bottom point.
 *
 * \internal Note that since the azimuthal region is periodic and can start
 * anywhere from zero to 1 turn, we have to make decisions about its shape
 * based on the stop angle rather than end angle, else we'd have to
 * restrict the input start value to `+/- pi` or something. In contrast, the
 * \em polar region is on a non-periodic range `[0, 0.5]` and we have to...
 */
EnclosedPolar::EnclosedPolar(Turn start, Turn stop)
    : start_{start}, stop_{stop}
{
    CELER_VALIDATE(start_ >= zero_quantity() && start_ < Turn{0.5},
                   << "invalid start angle " << start_.value()
                   << " [turns]: must be in [0, 0.5)");
    CELER_VALIDATE(stop_ > start_ && stop_ <= Turn{0.5},
                   << "invalid stop angle " << stop.value()
                   << " [turns]: must be in (" << start_.value() << ", 0.5]");
}

//---------------------------------------------------------------------------//
/*!
 * Construct one or two wedges to union.
 *
 * The result will be intersected the solid: these wedges are the parts to
 * \em keep.
 */
auto EnclosedPolar::make_regions() const -> VecPolarWedge
{
    CELER_EXPECT(*this);

    VecPolarWedge result;

    static constexpr Turn equator{0.25};

    if (start_ < equator)
    {
        result.emplace_back(start_, std::min(equator, stop_));
    }
    if (stop_ > equator)
    {
        result.emplace_back(std::max(equator, start_), stop_);
    }

    CELER_ENSURE(!result.empty());
    return result;
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
        char const* ext = (sense == Sense::outside ? "~azi" : "azi");
        NodeId wedge_id = build_intersect_region(vb, this->label(), ext, wedge);
        if (sense == Sense::outside)
        {
            wedge_id = vb.insert_region({}, Negated{wedge_id});
        }
        nodes.push_back(wedge_id);
    }

    if (auto const& pol = this->enclosed_polar())
    {
        // Union the polar wedge components
        std::vector<NodeId> wedge_nodes;
        for (auto const& wedge : pol.make_regions())
        {
            wedge_nodes.push_back(
                build_intersect_region(vb, this->label(), "pol", wedge));
        }
        auto union_id
            = vb.insert_region({}, Joined{op_or, std::move(wedge_nodes)});

        // Intersect the union with the result
        nodes.push_back(union_id);
    }

    // Intersect the given surfaces+regions to create a new CSG node
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
                        EnclosedAzi&& azi,
                        EnclosedPolar&& polar) -> SPConstObject
{
    if (!excluded && !azi && !polar)
    {
        // Just a shape
        return std::make_shared<Shape<T>>(std::move(label),
                                          std::move(interior));
    }

    return std::make_shared<Solid<T>>(std::move(label),
                                      std::move(interior),
                                      std::move(excluded),
                                      std::move(azi),
                                      std::move(polar));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optional excluded region and enclosed angle.
 */
template<class T>
Solid<T>::Solid(std::string&& label,
                T&& interior,
                OptionalRegion&& excluded,
                EnclosedAzi&& azi,
                EnclosedPolar&& polar)
    : label_{std::move(label)}
    , interior_{std::move(interior)}
    , exclusion_{std::move(excluded)}
    , azi_{std::move(azi)}
    , polar_{std::move(polar)}
{
    CELER_VALIDATE(exclusion_ || azi_ || polar_,
                   << "solid requires an excluded slice or region: use a "
                      "Shape instead");
    CELER_VALIDATE(!exclusion_ || interior_.encloses(*exclusion_),
                   << "solid '" << this->label()
                   << "' was given an interior region that is not enclosed by "
                      "its exterior");
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
