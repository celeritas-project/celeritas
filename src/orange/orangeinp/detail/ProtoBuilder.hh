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
        //! Save metadata during construction for each universe
        SaveUnivJson save_json;

        //! Remove bounding surfaces from lower-level universes
        bool implicit_parent_boundary{};
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

    // Get the UniverseId of the universe currently being built
    inline UnivId current_uid() const;

    //! Number of universes
    UnivId::size_type num_universes() const { return inp_->universes.size(); }

    //! Whether the current universe is the global universe
    bool is_global_universe() const
    {
        return this->current_uid() == orange_global_univ;
    }

    //! Whether to delete exterior boundaries of the current universe
    bool assume_inside() const
    {
        return implicit_parent_boundary_ && !this->is_global_universe();
    }

    //! Logic notation being constructed
    LogicNotation logic() const { return inp_->logic; }

    // Save debugging data for a universe
    void save_json(JsonPimpl&&) const;

    // Construct a universe (to be called *once* per proto)
    void insert(VariantUniverseInput&& unit);

  private:
    OrangeInput* inp_{};
    ProtoMap const& protos_;
    SaveUnivJson save_json_;
    bool implicit_parent_boundary_{};

    // State variables
    size_type num_univs_inserted_{0};
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
 * Get the UniverseId of the universe currently being built.
 */
UnivId ProtoBuilder::current_uid() const
{
    CELER_EXPECT(num_univs_inserted_ < this->num_universes());
    return UnivId{this->num_universes() - num_univs_inserted_ - 1};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
