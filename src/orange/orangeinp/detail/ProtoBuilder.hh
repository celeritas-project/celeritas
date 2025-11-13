//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ProtoBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"

#include "ProtoMap.hh"

namespace celeritas
{
struct JsonPimpl;
namespace orangeinp
{
class ProtoInterface;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage data and state during the universe construction.
 *
 * This is a helper class passed to UnitProto::build which manages data for the
 * UnitProto -> OrangeInput build process. It also maintains the universe ID
 * of the current universe being constructed.
 */
class ProtoBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    using SaveUnivJson = std::function<void(UnivId, JsonPimpl&&)>;
    //!@}

    //! Input options for construction
    struct Options
    {
        //! Manually specify a tracking/construction tolerance
        Tolerance<> tol;
        //! Save metadata during construction for each universe
        SaveUnivJson save_json;
    };

  public:
    // Construct with output pointer, geometry construction options, and protos
    ProtoBuilder(OrangeInput* inp, ProtoMap const& protos, Options const& opts);

    //! Get the tolerance to use when constructing geometry
    Tol const& tol() const { return inp_->tol; }

    //! Whether output should be saved for each
    bool save_json() const { return static_cast<bool>(save_json_); }

    // Find a universe ID
    inline UnivId find_universe_id(ProtoInterface const*) const;

    // Get the next universe ID
    inline UnivId next_id() const;

    // Get the bounding box of a universe
    inline BBox const& bbox(UnivId) const;

    // Expand the bounding box of a universe
    void expand_bbox(UnivId, BBox const& local_box);

    // Save debugging data for a universe
    void save_json(JsonPimpl&&) const;

    // Construct a universe (to be called *once* per proto)
    void insert(VariantUniverseInput&& unit);

    // The the UniverseId of the universe currently being built
    UnivId current_uid() const { return current_uid_; }

    // Whether or not the current universe is the global universe
    bool is_global_universe() const
    {
        return current_uid_ == orange_global_univ;
    }

  private:
    OrangeInput* inp_;
    ProtoMap const& protos_;
    SaveUnivJson save_json_;
    std::vector<BBox> bboxes_;

    // State variables
    UnivId current_uid_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find a universe ID.
 */
UnivId ProtoBuilder::find_universe_id(ProtoInterface const* p) const
{
    return protos_.find(p);
}

//---------------------------------------------------------------------------//
/*!
 * Get the bounding box of a universe.
 */
BBox const& ProtoBuilder::bbox(UnivId univ_id) const
{
    CELER_EXPECT(univ_id < bboxes_.size());
    return bboxes_[univ_id.get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
