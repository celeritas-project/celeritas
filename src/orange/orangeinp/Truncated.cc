//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Truncated.cc
//---------------------------------------------------------------------------//
#include "Truncated.hh"

#include "corecel/Assert.hh"
#include "corecel/io/JsonPimpl.hh"
#include "orange/orangeinp/IntersectRegion.hh"

#include "ObjectIO.json.hh"

#include "detail/BuildIntersectRegion.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
// Conveniently use build_intersect_region on a bunch of planes
class TruncatedRegion final : public IntersectRegionInterface
{
  public:
    using VecPlane = Truncated::VecPlane;

    explicit TruncatedRegion(VecPlane const& planes) : planes_(planes) {}

    void build(IntersectSurfaceBuilder& build_surface) const final
    {
        for (auto const& p : planes_)
        {
            p.build(build_surface);
        }
    }

    void output(JsonPimpl*) const final { CELER_ASSERT_UNREACHABLE(); }

  private:
    VecPlane const& planes_;
};
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with region to truncate and truncating planes.
 */
Truncated::Truncated(std::string&& label, UPRegion&& region, VecPlane&& planes)
    : label_{std::move(label)}
    , region_{std::move(region)}
    , planes_{std::move(planes)}
{
    CELER_EXPECT(region_);
    CELER_VALIDATE(!planes_.empty(),
                   << "truncated requires at least one truncated plane");

    // Sort planes by axis and sense for consistency
    auto as_tuple = [](Plane const& p) {
        return std::tuple(to_int(p.axis()), static_cast<bool>(p.sense()));
    };
    std::sort(planes_.begin(),
              planes_.end(),
              [as_tuple](Plane const& a, Plane const& b) {
                  return as_tuple(a) < as_tuple(b);
              });
    // Check for duplicates
    auto it = std::adjacent_find(planes_.begin(),
                                 planes_.end(),
                                 [as_tuple](Plane const& a, Plane const& b) {
                                     return as_tuple(a) == as_tuple(b);
                                 });

    CELER_VALIDATE(it == planes_.end(),
                   << "duplicate axis/sense combination in truncated region");
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId Truncated::build(VolumeBuilder& vb) const
{
    std::vector<NodeId> nodes;

    // Build the base region (interior)
    nodes.push_back(
        build_intersect_region(vb, this->label(), "interior", *region_));

    // Build the truncating planes
    nodes.push_back(build_intersect_region(
        vb, this->label(), "trunc", TruncatedRegion{planes_}));

    // Intersect the given surfaces to create a new CSG node
    return vb.insert_region(Label{std::string{this->label()}},
                            Joined{op_and, std::move(nodes)});
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void Truncated::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
