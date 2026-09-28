//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedGeoTrackView.cc
//---------------------------------------------------------------------------//
#include "CheckedGeoTrackView.hh"

#include <optional>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayOperators.hh"  // IWYU pragma: keep
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/SoftEqual.hh"

#include "GeoInterface.hh"
#include "GeoParamsInterface.hh"  // IWYU pragma: keep
#include "UnitLength.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//

std::ostream& full_precision(std::ostream& os)
{
    os.precision(CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT ? 7 : 14);
    return os;
}

//---------------------------------------------------------------------------//
//! Print a label with the native unit length
struct NativeLength
{
    friend std::ostream& operator<<(std::ostream& os, NativeLength const&)
    {
        os << " [" << native_unit_length.label << ']';
        return os;
    }
};

//---------------------------------------------------------------------------//

[[noreturn]] void throw_cgtv_error(CheckedGeoTrackView const& cgtv,
                                   std::ostringstream&& msg,
                                   std::string&& cond,
                                   char const* file,
                                   int line)
{
    msg << ": " << cgtv;
    throw CheckedGeoError{{RuntimeError::validate_err_str,
                           std::move(msg).str(),
                           std::move(cond),
                           file,
                           line}};
}

#define CGTV_FAIL(CGTV, WHERE, WHAT) \
    do \
    { \
        std::ostringstream msg_; \
        msg_ << "failed during " << WHERE; \
        msg_ << "" WHAT; \
        throw_cgtv_error(CGTV, std::move(msg_), {}, __FILE__, __LINE__); \
    } while (0)

#define CGTV_VALIDATE_NOT_FAILED(CGTV, WHERE) \
    do \
    { \
        if ((CGTV).check_failure() && CELER_UNLIKELY((CGTV).failed())) \
        { \
            CGTV_FAIL(CGTV, WHERE, ); \
        } \
    } while (0)

#define CGTV_VALIDATE(CGTV, COND, STREAM_WHAT) \
    do \
    { \
        if (CELER_UNLIKELY(!(COND))) \
        { \
            std::ostringstream msg_; \
            msg_ << "" STREAM_WHAT; \
            throw_cgtv_error( \
                CGTV, std::move(msg_), #COND, __FILE__, __LINE__); \
        } \
    } while (0)

#define CGTV_LOG(LEVEL) \
    this->log_(CELER_CODE_PROVENANCE, ::celeritas::LogLevel::LEVEL)
//---------------------------------------------------------------------------//
Logger default_checked_geo_logger()
{
    // Default to copying the self logger (including error handler and level)
    auto result = ::celeritas::self_logger();
    // Override log level with CELER_LOG_GEO
    result.level(getenv_loglevel("CELER_LOG_GEO", result.level()));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with a unique pointer to a geo track view.
 */
CheckedGeoTrackView::CheckedGeoTrackView(UPTrack track,
                                         SPConstVolumes volumes,
                                         SPConstGeoI geo_interface,
                                         UnitLength unit_length)
    : t_{std::move(track)}
    , log_{default_checked_geo_logger()}
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
CheckedGeoTrackView& CheckedGeoTrackView::operator=(
    GeoTrackInitializer const& init)
{
    CELER_EXPECT(t_);
    CGTV_LOG(debug) << "Initializing at " << full_precision
                    << StreamableLength{init.pos, unit_length_} << " along "
                    << init.dir;
    CELER_VALIDATE(is_soft_unit_vector(init.dir),
                   << "cannot initialize with a non-unit direction "
                   << repr(init.dir));

    *t_ = init;
    CGTV_VALIDATE_NOT_FAILED(*this, "initialization");
    CGTV_VALIDATE(*this, !t_->is_outside(), << "initialized outside");
    count_ = {};
    next_step_.reset();

    if (t_->is_on_boundary())
    {
        CGTV_LOG(warning) << "Started on a boundary: " << *this;
    }
    CGTV_LOG(status) << "Initialized " << *this;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Derive the geometry status from the inner track's state.
 */
GeoStatus CheckedGeoTrackView::geo_status() const
{
    return t_->geo_status();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance.
 *
 * \deprecated Infinite safety should be replaced with manual safety (REMOVE in
 * v0.8)
 * \return Nonnegative safety value
 */
real_type CheckedGeoTrackView::find_safety()
{
    return t_->find_safety(NumericLimits<real_type>::infinity());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance up to a given length.
 *
 * \return Nonnegative safety value up to max_safety
 */
real_type CheckedGeoTrackView::find_safety(real_type max_safety)
{
    CGTV_LOG(debug) << "Finding safety up to " << full_precision << max_safety;
    CELER_VALIDATE(max_safety > 0,
                   << "invalid safety maximum " << repr(max_safety)
                   << NativeLength{});
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");

    ++count_.safety;

    real_type result = t_->find_safety(max_safety);
    CGTV_VALIDATE_NOT_FAILED(*this, "find_safety");

    CGTV_VALIDATE(*this,
                  result >= 0,
                  << "invalid safety result " << repr(result)
                  << NativeLength{});

    if (result > max_safety)
    {
        CGTV_LOG(warning) << full_precision << "Returned safety "
                          << StreamableLength{result, unit_length_} << " at "
                          << StreamableLength{t_->pos(), unit_length_}
                          << " exceeds requested search distance "
                          << max_safety << NativeLength{};
    }
    else if (result == 0)
    {
        CGTV_LOG(warning) << "Encountered zero safety: " << *this;
    }
    else
    {
        CGTV_LOG(status) << "Found safety "
                         << StreamableLength{result, unit_length_} << " at "
                         << StreamableLength{t_->pos(), unit_length_};
    }

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
    CGTV_LOG(debug) << "Changing direction to " << newdir;
    CELER_VALIDATE(is_soft_unit_vector(newdir),
                   << "cannot change to a non-unit direction " << repr(newdir));
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(),
                   << "cannot change direction while outside");

    bool started_on_boundary = t_->is_on_boundary();
    ImplVolumeId impl_vol = t_->impl_volume_id();
    t_->set_dir(newdir);
    CGTV_VALIDATE_NOT_FAILED(*this, "set_dir");
    CGTV_VALIDATE(*this,
                  started_on_boundary == t_->is_on_boundary(),
                  << "boundary state changed during set_dir");
    CGTV_VALIDATE(*this,
                  impl_vol == t_->impl_volume_id(),
                  << "volume changed during set_dir");
    next_step_.reset();

    CGTV_LOG(status) << "Set direction to " << repr(t_->dir());
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
    CGTV_LOG(debug) << full_precision << "Finding next step up to " << distance;
    CELER_VALIDATE(distance > 0,
                   << "invalid step maximum " << repr(distance)
                   << NativeLength{});
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(),
                   << "cannot find next step from outside");
    auto const& units = this->unit_length();

    if (next_step_ && distance <= *next_step_)
    {
        CGTV_LOG(warning) << "Finding next step up to " << repr(distance)
                          << NativeLength{} << " when previous step "
                          << repr(*next_step_) << NativeLength{}
                          << " was already calculated";
    }

    bool const started_on_boundary{t_->is_on_boundary()};
    ++count_.intersect;
    auto result = t_->find_next_step(distance);
    CGTV_VALIDATE_NOT_FAILED(*this, "find_next_step");
    if (check_next_safety_ && result.boundary
        && result.distance > this->safety_tol() && !started_on_boundary)
    {
        real_type safety = t_->find_safety(distance);
        if (!(safety <= result.distance || soft_equal(safety, result.distance)))
        {
            CGTV_LOG(warning)
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
    if (result.distance == 0)
    {
        if (check_zero_distance_)
        {
            // TODO: replace zero-distance from reentering geometry (ORANGE and
            // VecGeom 2+) with a different propagation status
            CGTV_LOG(info) << "Returning zero distance: " << *this;
        }
        if (t_->is_on_boundary() != started_on_boundary)
        {
            CGTV_LOG(warning)
                << "find_next_step changed boundary state: new status is "
                << t_->geo_status();
        }
    }
    CGTV_VALIDATE(*this,
                  result.distance >= 0 && result.distance <= distance,
                  << "return distance " << repr(result.distance)
                  << NativeLength{} << " out of bounds " << repr(distance)
                  << NativeLength{});
    CGTV_VALIDATE(*this,
                  t_->is_on_boundary() == started_on_boundary
                      || result.distance == 0,
                  << "boundary state changed during find_next_step (started "
                  << (started_on_boundary ? "on" : "off") << " boundary)");
    CGTV_LOG(status) << (result.boundary ? "Found" : "No") << " boundary at "
                     << StreamableLength{result.distance, unit_length_};

    if (result.boundary || result.distance > next_step_.value_or(0_r))
    {
        next_is_boundary_ = result.boundary;
        next_step_ = result.distance;
    }
    CELER_ENSURE(next_step_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume along the current direction.
 *
 * \pre Boundary must have been found and \em step is less than it
 * \post Not on boundary
 */
void CheckedGeoTrackView::move_internal(real_type step)
{
    CGTV_LOG(debug) << "Moving " << full_precision
                    << StreamableLength{step, unit_length_};
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "cannot move while outside");
    CELER_VALIDATE(next_step_, << "tried to move before finding the next step");
    CELER_VALIDATE(step <= *next_step_,
                   << "internal step " << step << " exceeds linear step "
                   << *next_step_ << " by " << (step - *next_step_)
                   << NativeLength{});
    CELER_VALIDATE(step != next_step_ || !next_is_boundary_,
                   << "cannot move_internal to a boundary");

    t_->move_internal(step);
    *next_step_ -= step;
    if (next_step_ <= 0)
    {
        next_step_.reset();
    }
    CGTV_VALIDATE_NOT_FAILED(*this, "move_internal");
    CGTV_VALIDATE(*this,
                  !t_->is_on_boundary() && !t_->is_outside(),
                  << "on boundary after moving " << repr(step)
                  << NativeLength{});
    CGTV_LOG(status) << "Moved internally by "
                     << StreamableLength{step, unit_length_};
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume to a nearby position.
 *
 * The first call to this function will perform additional checking by
 * reinitializing the geometry at the given position.
 *
 * \note We do not validate that the input position is path-connected with the
 * current position since that's non-trivial.
 *
 * \pre Inside the geometry
 * \post Not on boundary
 */
void CheckedGeoTrackView::move_internal(Real3 const& pos)
{
    CGTV_LOG(debug) << "Moving to " << full_precision
                    << StreamableLength{pos, unit_length_};
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "cannot move while outside");
    // TODO: store and check last found safety

    real_type orig_safety = (t_->is_on_boundary() ? 0 : t_->find_safety());
    auto orig_pos = t_->pos();
    t_->move_internal(pos);
    next_step_.reset();
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
            CELER_LOG_LOCAL(warning)
                << "Moved internally from boundary but safety didn't "
                   "increase: volume "
                << t_->impl_volume_id().get() << " from " << repr(orig_pos)
                << " to " << repr(t_->pos())
                << " (distance: " << distance(orig_pos, pos) << NativeLength{}
                << ")";
        }
    }
    CGTV_LOG(status) << "Moved internally to "
                     << StreamableLength{pos, unit_length_};
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 *
 * \pre Boundary has been found
 * \post On boundary
 */
void CheckedGeoTrackView::move_to_boundary()
{
    CGTV_LOG(debug) << "Moving to boundary";
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "invalid call while outside");
    CELER_VALIDATE(next_step_ && next_is_boundary_,
                   << "invalid call while outside");
    auto const step = *next_step_;

    t_->move_to_boundary();
    CGTV_VALIDATE_NOT_FAILED(*this, "move_to_boundary");
    checked_internal_ = false;
    next_step_.reset();
    next_is_boundary_ = false;

    CGTV_VALIDATE(*this,
                  t_->is_on_boundary(),
                  << "moving to boundary did not leave track on a boundary");
    CGTV_LOG(status) << "Moved to boundary at "
                     << StreamableLength{step, unit_length_};
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
void CheckedGeoTrackView::cross_boundary()
{
    CGTV_LOG(debug) << "Crossing boundary";
    CELER_VALIDATE(!this->failed() || !check_failure_, << "failure exists");
    CELER_VALIDATE(!this->is_outside(), << "invalid call while outside");

    CELER_VALIDATE(t_->is_on_boundary(),
                   << "cannot cross boundary without being on boundary");

    // Capture pre-crossing normal if checking is enabled
    std::optional<Real3> pre_norm;
    if (check_normal_ && !t_->is_outside())
    {
        pre_norm = t_->normal();
    }

    // Cross boundary
    t_->cross_boundary();
    CGTV_VALIDATE_NOT_FAILED(*this, "cross_boundary");
    CGTV_VALIDATE(*this,
                  t_->is_on_boundary(),
                  << "not on boundary after crossing boundary");

    // Verify post-crossing normal if checking is enabled
    std::optional<Real3> post_norm;
    if (pre_norm && !t_->is_outside())
    {
        SoftZero const soft_zero_dir{celeritas::sqrt_tol<real_type>};
        post_norm = t_->normal();
        auto norm_error = min(std::fabs(distance(*pre_norm, *post_norm)),
                              std::fabs(distance(*pre_norm, -*post_norm)));
        CGTV_VALIDATE(*this,
                      soft_zero_dir(norm_error),
                      << "inconsistent surface normal "
                      << "(error " << norm_error << "): pre-crossing "
                      << repr(*pre_norm) << ", post-crossing "
                      << repr(*post_norm));
        constexpr real_type tight_tol
            = std::is_same_v<real_type, float> ? 1e-5 : 1e-10;
        if (!SoftZero<real_type>{tight_tol}(norm_error))
        {
            CGTV_LOG(warning)
                << "Pre- and post- surface normals almost disagree "
                << "(error " << norm_error << "): pre-crossing "
                << repr(*pre_norm) << ", post-crossing " << repr(*post_norm);
        }

        // Check for tangent crossing
        auto dot_normal = dot_product(t_->dir(), *post_norm);
        if (soft_zero_dir(dot_normal))
        {
            CGTV_LOG(warning)
                << "Crossed at a tangent: dot product of exiting normal "
                << repr(*post_norm) << " with direction " << repr(t_->dir())
                << " is " << dot_normal;
        }
    }
    {
        auto msg = CGTV_LOG(status);
        msg << "Crossed boundary";
        if (pre_norm)
        {
            msg << " with pre-crossing normal " << *pre_norm;
        }
        if (this->volumes())
        {
            msg << " into " << StreamableUniqueVolName{*this, *this->volumes()};
        }
        if (post_norm)
        {
            msg << " with post-crossing normal " << *post_norm;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Output the state of a checked track view.
 */
std::ostream& operator<<(std::ostream& os, CheckedGeoTrackView const& geo)
{
    // Print high-precision pos/dir with desired units
    auto const& units = geo.unit_length();
    auto const orig_precision = os.precision();
    os << full_precision << "at " << StreamableLength{geo.pos(), units}
       << " along " << geo.dir() << ", ";
    os.precision(orig_precision);

    // Flags and states
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
}  // namespace celeritas
