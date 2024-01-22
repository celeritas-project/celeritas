//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/LabelIdMultiMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/io/Label.hh"

#include "Range.hh"
#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Map IDs to label+sublabel.
 *
 * Many Geant4 geometry definitions reuse a label for material or volume
 * definitions, and we need to track the unique name across multiple codes
 * (Geant4, VecGeom+GDML) to differentiate between materials or volumes. This
 * class maps a "label" to a range of "sublabels", each of which corresponds to
 * a unique ID. It also provides the reverse mapping so that an ID can retrieve
 * the corresponding label/sublabel pair.
 *
 * There is no requirement that sublabels be ordered adjacent to each other:
 * the IDs corresponding to a label may be noncontiguous.
 *
 * Duplicate labels are allowed but will be added to a list of duplicate IDs
 * that can be warned about downstream. Empty labels will be ignored.
 *
 * If no sublabels or labels are available, an empty span or "false" OpaqueId
 * will be returned.
 */
template<class I>
class LabelIdMultiMap
{
  public:
    //!@{
    //! \name Type aliases
    using IdT = I;
    using SpanConstIdT = Span<IdT const>;
    using VecLabel = std::vector<Label>;
    using size_type = typename IdT::size_type;
    //!@}

  public:
    // Empty constructor for delayed build
    LabelIdMultiMap() = default;

    // Construct from a vector of label+sublabel pairs
    explicit LabelIdMultiMap(VecLabel keys);

    // Access the range of IDs corresponding to a label
    inline SpanConstIdT find_all(std::string const& label) const;

    // Access an ID by label/sublabel pair
    inline IdT find(Label const& label_sub) const;

    // Access the label+sublabel pair for an Id
    inline Label const& get(IdT id) const;

    //! Get duplicate labels to warn about
    SpanConstIdT duplicates() const { return make_span(duplicates_); }

    //! Get the number of elements
    size_type size() const { return keys_.size(); }

  private:
    VecLabel keys_;
    std::vector<IdT> id_data_;
    std::vector<size_type> id_offsets_;
    std::vector<IdT> duplicates_;
    std::unordered_map<std::string, size_type> ids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<class I>
LabelIdMultiMap<I>::LabelIdMultiMap(VecLabel keys) : keys_(std::move(keys))
{
    if (keys_.empty())
    {
        // Sometimes we don't have any items to map. Rely on the
        // default-constructed values.
        return;
    }

    // Build list of IDs corresponding to each key
    id_data_.resize(keys_.size());
    for (auto idx : range<size_type>(keys_.size()))
    {
        id_data_[idx] = IdT{idx};
    }
    // Reorder consecutive ID data based on string lexicographic ordering
    std::sort(id_data_.begin(), id_data_.end(), [this](IdT a, IdT b) {
        return this->keys_[a.unchecked_get()] < this->keys_[b.unchecked_get()];
    });

    // Reserve space for groups
    id_offsets_.reserve(keys_.size() / 2 + 2);
    ids_.reserve(id_offsets_.capacity());

    // Search for when the label changes
    id_offsets_.push_back(0);
    ids_.insert({keys_[id_data_[0].unchecked_get()].name, 0});
    for (auto idx : range<size_type>(1, id_data_.size()))
    {
        Label const& prev = keys_[id_data_[idx - 1].unchecked_get()];
        Label const& cur = keys_[id_data_[idx].unchecked_get()];
        if (prev == cur && !cur.empty())
        {
            if (duplicates_.empty()
                || keys_[duplicates_.back().unchecked_get()] != prev)
            {
                // Push back previous entry if it's not already there
                duplicates_.push_back(id_data_[idx - 1]);
            }
            duplicates_.push_back(id_data_[idx]);
        }
        if (prev.name != cur.name)
        {
            // Add start index of the new name
            size_type offset_idx = id_offsets_.size();
            id_offsets_.push_back(idx);
            auto insert_ok = ids_.insert({cur.name, offset_idx});
            CELER_ASSERT(insert_ok.second);
        }
    }
    id_offsets_.push_back(id_data_.size());

    CELER_ENSURE(keys_.size() == id_data_.size());
    CELER_ENSURE(id_offsets_.size() == ids_.size() + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Access the range of IDs corresponding to a label.
 */
template<class I>
auto LabelIdMultiMap<I>::find_all(std::string const& name) const
    -> SpanConstIdT
{
    auto iter = ids_.find(name);
    if (iter == ids_.end())
        return {};

    size_type offset_idx = iter->second;
    CELER_ASSERT(offset_idx + 1 < id_offsets_.size());
    size_type start = id_offsets_[offset_idx];
    size_type stop = id_offsets_[offset_idx + 1];
    CELER_ENSURE(start < stop && stop <= id_data_.size());
    return {id_data_.data() + start, stop - start};
}

//---------------------------------------------------------------------------//
/*!
 * Access an ID by label/sublabel pair.
 *
 * This returns a \c false OpaqueId if no such label pair exists.
 */
template<class I>
auto LabelIdMultiMap<I>::find(Label const& label_sub) const -> IdT
{
    auto items = this->find_all(label_sub.name);

    // Just do a linear search on sublabels
    auto iter
        = std::find_if(items.begin(), items.end(), [this, &label_sub](IdT id) {
              CELER_EXPECT(id < this->keys_.size());
              return this->keys_[id.unchecked_get()].ext == label_sub.ext;
          });
    if (iter == items.end())
    {
        // No sublabel matches
        return {};
    }
    CELER_ENSURE(keys_[iter->unchecked_get()] == label_sub);
    return *iter;
}

//---------------------------------------------------------------------------//
/*!
 * Access the label+sublabel pair for an Id.
 *
 * This raises an exception if the ID is outside of the valid range.
 */
template<class I>
Label const& LabelIdMultiMap<I>::get(IdT id) const
{
    CELER_EXPECT(id < this->size());
    return keys_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
