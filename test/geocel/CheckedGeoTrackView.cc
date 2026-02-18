//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedGeoTrackView.cc
//---------------------------------------------------------------------------//
#include "CheckedGeoTrackView.hh"

#include <optional>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/VolumeParams.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//

struct StreamableUniqueVolName
{
    GeoTrackInterface<real_type> const& geo;
    VolumeParams const& params;
};

std::ostream& operator<<(std::ostream& os, StreamableUniqueVolName const& suvn)
{
    if (suvn.geo.is_outside())
    {
        return os << "[OUTSIDE]";
    }

    auto const& vi_labels = suvn.params.volume_instance_labels();
    if (vi_labels.empty())
    {
        return os;
    }

    auto vlev = suvn.geo.volume_level();
    CELER_ASSERT(vlev && vlev >= VolumeLevelId{0});

    std::vector<VolumeInstanceId> ids(vlev.get() + 1);
    suvn.geo.volume_instance_id(make_span(ids));

    os << vi_labels.at(ids[0]);
    for (auto i : range(std::size_t{1}, ids.size()))
    {
        os << '/';
        if (ids[i])
        {
            os << vi_labels.at(ids[i]);
        }
        else
        {
            os << "[INVALID]";
        }
    }
    return os;
}

//! Print a length/position as a quantity with units
template<class T>
struct StreamableLength
{
    T const& native_value;
    UnitLength const& units;
};

// Needed for C++17
template<class T>
StreamableLength(T const&, UnitLength) -> StreamableLength<T>;

template<class T>
std::ostream& operator<<(std::ostream& os, StreamableLength<T> const& sl)
{
    os << repr(sl.units.from_native(sl.native_value)) << " [" << sl.units.label
       << ']';
    return os;
}

//! Print a length/position as a quantity with units
struct NativeLength
{
};

std::ostream& operator<<(std::ostream& os, NativeLength const&)
{
    os << " [" << lengthunits::native_label << ']';
    return os;
}

//---------------------------------------------------------------------------//

[[noreturn]] void throw_cgtv_error(CheckedGeoTrackView const& cgtv,
                                   std::ostringstream&& msg,
                                   std::string&& cond,
                                   char const* file,
                                   int line)
{
    msg << ": " << cgtv;
    throw CheckedGeoError{
        {RuntimeError::validate_err_str, std::move(msg).str(), cond, file, line}};
}

#define CGTV_VALIDATE_NOT_FAILED(CGTV, WHERE)                                \
    do                                                                       \
    {                                                                        \
        if ((CGTV).check_failure() && CELER_UNLIKELY((CGTV).failed()))       \
        {                                                                    \
            std::ostringstream msg_;                                         \
            msg_ << "failed during " << WHERE;                               \
            throw_cgtv_error(CGTV, std::move(msg_), {}, __FILE__, __LINE__); \
        }                                                                    \
    } while (0)

#define CGTV_VALIDATE(CGTV, COND, WHAT)                            \
    do                                                             \
    {                                                              \
        if (CELER_UNLIKELY(!(COND)))                               \
        {                                                          \
            std::ostringstream msg_;                               \
            msg_ WHAT;                                             \
            throw_cgtv_error(                                      \
                CGTV, std::move(msg_), #COND, __FILE__, __LINE__); \
        }                                                          \
    } while (0)

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * ! Construct with a unique pointer to a geo track view.
 */
CheckedGeoTrackView::CheckedGeoTrackView(UPTrack track,
                                         SPConstVolumes volumes,
                                         SPConstGeoI geo_interface,
                                         UnitLength unit_length)
    : t_{std::move(track)}
    , volumes_{std::move(volumes)}
    , geo_interface_{std::move(geo_interface)}
    , unit_length_(unit_length)
{
    CELER_EXPECT(unit_length_.value > 0);
    if (geo_interface_)
    {
        check_safety_ = geo_interface_->supports_safety();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the state.
 */
CheckedGeoTrackView&
CheckedGeoTrackView::operator=(GeoTrackInitializer const& init)
{
    CELER_EXPECT(t_);
    CELER_VALIDATE(is_soft_unit_vector(init.dir),
                   << "cannot initialize with a non-unit direction "
                   << repr(init.dir));

    *t_ = init;
    CGTV_VALIDATE_NOT_FAILED(*this, "initialization");
    CGTV_VALIDATE(*this, !t_->is_outside(), << "initialized outside");
    if (t_->is_on_boundary())
    {
        CELER_LOG_LOCAL(warning) << "Started on a boundary: " << *this;
    }
    count_ = {};
    next_boundary_.reset();
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance.
 *
 * \return Nonnegative safety value
 */
real_type CheckedGeoTrackView::find_safety()
{
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_on_boundary(),
                   << "cannot find safety when on a boundary");

    ++count_.safety;

    auto result = t_->find_safety();
    CGTV_VALIDATE_NOT_FAILED(*this, "find_safety");
    CGTV_VALIDATE(*this,
                  result >= 0,
                  << "safety " << repr(result) << NativeLength{}
                  << " is out of bounds");
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance up to a given length.
 *
 * \return Nonnegative safety value up to max_safety
 */
real_type CheckedGeoTrackView::find_safety(real_type max_safety)
{
    CELER_VALIDATE(max_safety > 0,
                   << "invalid safety maximum " << repr(max_safety)
                   << NativeLength{});
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");

    ++count_.safety;

    real_type result = t_->find_safety(max_safety);
    CGTV_VALIDATE_NOT_FAILED(*this, "find_safety");
    CGTV_VALIDATE(*this,
                  result >= 0 && result <= max_safety,
                  << "safety " << repr(result) << NativeLength{}
                  << " is out of bounds: should be in [0, " << max_safety
                  << ']');
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set the direction.
 *
 * \pre Direction is a unit vector
 * \post Boundary state was unaffected
 */
void CheckedGeoTrackView::set_dir(Real3 const& newdir)
{
    CELER_VALIDATE(is_soft_unit_vector(newdir),
                   << "cannot change to a non-unit direction " << repr(newdir));
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(),
                   << "cannot change direction while outside");

    bool started_on_boundary = t_->is_on_boundary();
    t_->set_dir(newdir);
    CGTV_VALIDATE_NOT_FAILED(*this, "set_dir");
    CGTV_VALIDATE(*this,
                  started_on_boundary == t_->is_on_boundary(),
                  << "boundary state changed during set_dir");
    next_boundary_.reset();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next boundary.
 *
 * \pre Cannot call from outside or if failed, distance is positive
 * \post Boundary state was unaffected
 * \return Next step, with distance between zero and the given maximum
 */
Propagation CheckedGeoTrackView::find_next_step(real_type distance)
{
    CELER_VALIDATE(distance > 0,
                   << "invalid step maximum " << repr(distance)
                   << NativeLength{});
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(),
                   << "cannot find next step from outside");
    auto const& units = this->unit_length();

    if (next_boundary_ && distance <= *next_boundary_)
    {
        CELER_LOG_LOCAL(warning)
            << "Finding next step up to " << repr(distance) << NativeLength{}
            << " when previous step " << repr(*next_boundary_)
            << NativeLength{} << " was already calculated";
    }

    bool const started_on_boundary{t_->is_on_boundary()};
    ++count_.intersect;
    auto result = t_->find_next_step(distance);
    CGTV_VALIDATE_NOT_FAILED(*this, "find_next_step");
    if (check_next_safety_ && result.boundary
        && result.distance > this->safety_tol() && !started_on_boundary)
    {
        real_type safety = t_->find_safety(distance);
        if (!(safety <= result.distance))
        {
            CELER_LOG_LOCAL(warning)
                << "Calculated safety " << safety << NativeLength{}
                << " exceeds actual distance " << result.distance
                << NativeLength{} << " to boundary at " << t_->pos()
                << NativeLength{} << " by " << safety - result.distance
                << NativeLength{} << ": " << *this;
            CGTV_VALIDATE(*this,
                          safety <= 1.1 * result.distance,
                          << "calculated safety "
                          << (StreamableLength{safety, units})
                          << " is much too large");
        }
    }
    if (check_zero_distance_ && result.distance == 0)
    {
        // TODO: replace zero-distance from reentering geometry (ORANGE)
        // with a different propagation status
        CELER_LOG_LOCAL(warning)
            << "Returning zero distance should be prohibited: " << *this;
    }
    CGTV_VALIDATE(*this,
                  result.distance >= 0 && result.distance <= distance,
                  << "return distance " << repr(result.distance)
                  << NativeLength{} << " out of bounds " << repr(distance)
                  << NativeLength{});
    CGTV_VALIDATE(*this,
                  t_->is_on_boundary() == started_on_boundary,
                  << "boundary state changed during find_next_step (started "
                  << (started_on_boundary ? "on" : "off") << " boundary)");

    next_boundary_ = result.distance;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume along the current direction.
 *
 * \post Not on boundary
 */
void CheckedGeoTrackView::move_internal(real_type step)
{
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "cannot move while outside");
    // TODO: check next_boundary_

    t_->move_internal(step);
    next_boundary_.reset();
    CGTV_VALIDATE_NOT_FAILED(*this, "move_internal");
    CGTV_VALIDATE(*this,
                  !t_->is_on_boundary() && !t_->is_outside(),
                  << "on boundary after moving " << repr(step)
                  << NativeLength{});
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume to a nearby position.
 *
 * The first call to this function will perform additional checking by
 * reinitializing the geometry at the given position.
 *
 * \post Not on boundary
 */
void CheckedGeoTrackView::move_internal(Real3 const& pos)
{
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "cannot move while outside");

    real_type orig_safety = (t_->is_on_boundary() ? 0 : t_->find_safety());
    auto orig_pos = t_->pos();
    t_->move_internal(pos);
    next_boundary_.reset();
    CGTV_VALIDATE_NOT_FAILED(*this, "move_internal");
    CGTV_VALIDATE(*this,
                  !this->is_on_boundary() && !t_->is_outside(),
                  << "not internal to volume after moving to " << repr(pos)
                  << NativeLength{});
    if (!checked_internal_ && orig_safety > this->safety_tol())
    {
        ImplVolumeId expected = t_->impl_volume_id();
        Initializer_t here{t_->pos(), t_->dir()};
        *t_ = here;
        CGTV_VALIDATE(*this,
                      t_->impl_volume_id() == expected,
                      << "volume ID changed during internal move from "
                      << repr(orig_pos) << NativeLength{} << ": was "
                      << expected.get() << ", now "
                      << t_->impl_volume_id().get());
        checked_internal_ = true;
    }
    if (check_safety_ && orig_safety == 0 && !t_->is_on_boundary())
    {
        real_type new_safety = t_->find_safety();
        if (!(new_safety > 0))
        {
            auto const& units = this->unit_length();
            CELER_LOG_LOCAL(warning)
                << "Moved internally from boundary but safety didn't "
                   "increase: volume "
                << t_->impl_volume_id().get() << " from " << repr(orig_pos)
                << NativeLength{} << " to " << repr(t_->pos())
                << NativeLength{} << " (distance: " << distance(orig_pos, pos)
                << NativeLength{} << ")";
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 *
 * \post On boundary
 */
void CheckedGeoTrackView::move_to_boundary()
{
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "invalid call while outside");

    // Move to boundary
    t_->move_to_boundary();
    CGTV_VALIDATE_NOT_FAILED(*this, "move_to_boundary");
    checked_internal_ = false;

    CGTV_VALIDATE(*this,
                  t_->is_on_boundary(),
                  << "moving to boundary did not leave track on a boundary");
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
void CheckedGeoTrackView::cross_boundary()
{
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "invalid call while outside");

    CELER_VALIDATE(t_->is_on_boundary(),
                   << "cannot cross boundary without being on boundary");

    // Capture pre-crossing normal if checking is enabled
    std::optional<Real3> pre_crossing_normal;
    if (check_normal_ && !t_->is_outside())
    {
        pre_crossing_normal = t_->normal();
    }

    // Cross boundary
    t_->cross_boundary();
    CGTV_VALIDATE_NOT_FAILED(*this, "cross_boundary");
    CGTV_VALIDATE(*this,
                  t_->is_on_boundary(),
                  << "not on boundary after crossing boundary");

    // Verify post-crossing normal if checking is enabled
    if (pre_crossing_normal && !t_->is_outside())
    {
        auto post_norm = t_->normal();
        CGTV_VALIDATE(
            *this,
            soft_equal(std::fabs(dot_product(*pre_crossing_normal, post_norm)),
                       real_type{1}),
            << "inconsistent surface normal: pre-crossing "
            << *pre_crossing_normal << ", post-crossing " << post_norm);

        // Check for tangent crossing warning
        if (soft_zero(dot_product(t_->dir(), post_norm)))
        {
            CELER_LOG_LOCAL(warning)
                << "Crossed at a tangent normal " << repr(post_norm)
                << ": post-crossing state is " << *this;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust volume name based on the geo state.
 */
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        VolumeParams const& params)
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto const& vol_labels = params.volume_labels();
    if (vol_labels.empty())
        return {};

    VolumeId id = geo.volume_id();
    if (!(id < vol_labels.size()))
    {
        return "[INVALID]";
    }

    return vol_labels.at(id).name;
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust impl volume name based on the geo state.
 */
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        GeoParamsInterface const& params)
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto const& vol_labels = params.impl_volumes();
    if (vol_labels.empty())
        return {};

    ImplVolumeId id = geo.impl_volume_id();
    if (!(id < vol_labels.size()))
    {
        return "[INVALID]";
    }

    return vol_labels.at(id).name;
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust volume instance name based on the geo state.
 */
std::string volume_instance_name(GeoTrackInterface<real_type> const& geo,
                                 VolumeParams const& params)
{
    if (geo.is_outside())
    {
        return "[OUTSIDE]";
    }

    auto const& vi_labels = params.volume_instance_labels();
    if (vi_labels.empty())
        return {};

    VolumeInstanceId vi_id;
    try
    {
        vi_id = geo.volume_instance_id();
    }
    catch (celeritas::DebugError const& e)
    {
        std::ostringstream os;
        auto const& d = e.details();
        os << "<exception at " << d.file << ':' << d.line << ": "
           << d.condition << '>';
        return std::move(os).str();
    }
    if (!(vi_id < vi_labels.size()))
    {
        return "[INVALID]";
    }

    return to_string(vi_labels.at(vi_id));
}

//---------------------------------------------------------------------------//
/*!
 * Get the descriptive, robust volume instance name based on the geo state.
 */
std::string unique_volume_name(GeoTrackInterface<real_type> const& geo,
                               VolumeParams const& params)
{
    std::ostringstream os;
    os << StreamableUniqueVolName{geo, params};
    return std::move(os).str();
}

//---------------------------------------------------------------------------//
/*!
 * Output the state of a checked track view.
 */
std::ostream& operator<<(std::ostream& os, CheckedGeoTrackView const& geo)
{
    // Length scale and description
    auto const& units = geo.unit_length();

    os << "at " << StreamableLength{geo.pos(), units} << " along "
       << repr(geo.dir()) << ", ";
    if (geo.failed())
    {
        os << "[FAILED] ";
    }
    if (geo.is_on_boundary())
    {
        os << "[ON BOUNDARY] ";
    }
    if (geo.volumes())
    {
        os << "in " << StreamableUniqueVolName{geo, *geo.volumes()};
    }
    else if (geo.geo_interface())
    {
        os << "in " << volume_name(geo, *geo.geo_interface());
    }
    else if (geo.is_outside())
    {
        os << "[OUTSIDE]";
    }
    else
    {
        // Unlikely/impossible
        os << "in impl volume " << geo.impl_volume_id().unchecked_get();
    }

    return os;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
