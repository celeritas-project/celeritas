//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * class maps a "label" to a name plus a range of "extensions", each of which
 * corresponds to a unique ID. It also provides the reverse mapping so that an
 * ID can retrieve the corresponding name/extension pair.
 *
 * There is no requirement that sublabels be ordered adjacent to each other:
 * the IDs corresponding to a label may be noncontiguous.
 *
 * Duplicate labels are allowed but will be added to a list of duplicate IDs
 * that can be warned about downstream. Empty labels will be ignored.
 *
 * If no sublabels or labels are available for a \c find_X call, an empty span
 * or "false" OpaqueId will be returned.
 *
 * The three kinds of \c find methods are named differently to avoid ambiguity:
 * - \c find_all returns the full set of IDs that match the given name;
 * - \c find_unique is a convenience accessor for locating a volume by name,
 *   but it only works if there are no duplicates; and
 * - \c find_exact looks for the full label, both name and extension.
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

    // Construct from a vector of label+sublabel pairs, with a type string
    inline LabelIdMultiMap(std::string&& label, VecLabel&& keys);

    // Construct from a vector of label+sublabel pairs, with no type
    inline explicit LabelIdMultiMap(VecLabel&& keys);

    // Access the range of IDs corresponding to a name
    inline SpanConstIdT find_all(std::string const& name) const;

    // Find an ID by name, throwing if not unique
    inline IdT find_unique(std::string const& name) const;

    // Access an ID by name/extension pair
    inline IdT find_exact(Label const& label) const;

    // Access the label+sublabel pair for an Id
    inline Label const& at(IdT id) const;

    //! Get the number of elements
    CELER_FORCEINLINE size_type size() const { return keys_.size(); }

    //! Whether no elements are present
    CELER_FORCEINLINE bool empty() const { return keys_.empty(); }

    // Whether this map is initialized
    inline explicit operator bool() const;

    //! Get duplicate labels to warn about
    SpanConstIdT duplicates() const { return make_span(duplicates_); }

  private:
    std::string type_label_;
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
 * Construct from a vector of label+sublabel pairs, with no type.
 */
template<class I>
LabelIdMultiMap<I>::LabelIdMultiMap(VecLabel&& keys)
    : LabelIdMultiMap{{}, std::move(keys)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a vector of label+sublabel pairs, with a type string.
 */
template<class I>
LabelIdMultiMap<I>::LabelIdMultiMap(std::string&& label, VecLabel&& keys)
    : type_label_{std::move(label)}, keys_{std::move(keys)}
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
    std::sort(id_data_.begin(), id_data_.end(), [&keys = keys_](IdT a, IdT b) {
        return keys[a.unchecked_get()] < keys[b.unchecked_get()];
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
 *
 * This is useful for identifiers that may be repeated in a problem definition
 * with uniquifying "extensions", such as pointer addresses from Geant4.
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
 * Find the ID corresponding to a label if exactly one exists.
 *
 * This will return an invalid ID if no labels match the given name, and it
 * will raise an exception if multiple labels do.
 */
template<class I>
auto LabelIdMultiMap<I>::find_unique(std::string const& name) const -> IdT
{
    auto items = this->find_all(name);
    if (items.empty())
        return {};
    CELER_VALIDATE(items.size() == 1,
                   << type_label_ << " '" << name << "' is not unique");
    return items.front();
}

//---------------------------------------------------------------------------//
/*!
 * Access an ID by exact label (name plus extension).
 *
 * This returns a \c false OpaqueId if no such label pair exists.
 */
template<class I>
auto LabelIdMultiMap<I>::find_exact(Label const& label_sub) const -> IdT
{
    auto items = this->find_all(label_sub.name);

    // Just do a linear search on sublabels
    auto iter = std::find_if(
        items.begin(), items.end(), [&keys = keys_, &label_sub](IdT id) {
            CELER_EXPECT(id < keys.size());
            return keys[id.unchecked_get()].ext == label_sub.ext;
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
Label const& LabelIdMultiMap<I>::at(IdT id) const
{
    CELER_EXPECT(id < this->size());
    return keys_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Whether this map is initialized.
 */
template<class I>
CELER_FORCEINLINE LabelIdMultiMap<I>::operator bool() const
{
    return !keys_.empty() || !type_label_.empty();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
