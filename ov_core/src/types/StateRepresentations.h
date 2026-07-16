#pragma once

#include <string>

namespace ov_type {
    enum class PoseStateRepresentation {
        LieGroupLeft,
        LieGroupRight,
        DecoupledRight,
        DecoupledLeft
    };

    inline std::string pose_rep_to_string(PoseStateRepresentation rep) {
        if (rep == PoseStateRepresentation::LieGroupLeft) {
            return "liegroupleft";
        } 
        return "";
    }
}; // namespace ov_type