#pragma once

#include "types/Type.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include "lie_utils/SO3.h"

namespace ov_type {

enum class PoseStateRepresentation { SE3Left, SE3Right, DecoupledRight, DecoupledLeft };

/**
 * @brief An SE3 Posetype that allows for multiple possible state representations.
 */
class PoseSE3 : public Type {
public:
  PoseSE3(PoseStateRepresentation pose_state_representation = PoseStateRepresentation::DecoupledRight)
      : Type(6), _pose_state_representation(pose_state_representation) {
    // Initialize subvariables
    Eigen::Matrix<double, 7, 1> pose0;
    pose0.setZero();
    Eigen::Matrix3d C_ab = Eigen::Matrix3d::Identity();
    Eigen::Quaterniond q_ab{C_ab};
    pose0(0, 0) = q_ab.x();
    pose0(1, 0) = q_ab.y();
    pose0(2, 0) = q_ab.z();
    pose0(3, 0) = q_ab.w();

    set_value(pose0);
    set_fej(pose0);
  }

  void set_local_id(int new_id) override { _id = new_id; }

  void update(const Eigen::VectorXd &dx) override {
    // Convert value to DCM and position
    Eigen::Quaterniond q_ab{_value.head<4>()};
    Eigen::Matrix3d C_ab = q_ab.toRotationMatrix();
    Eigen::Vector3d pos = _value.tail<3>();

    Eigen::Matrix3d C_ab_new;
    Eigen::Vector3d pos_new;
    if (_pose_state_representation == PoseStateRepresentation::DecoupledRight) {
      Eigen::Matrix3d C_ab_new = C_ab * ov_core::SO3::expMap(dx.head<3>());
      Eigen::Vector3d pos_new = pos + dx.tail<3>();
    } else {
      PRINT_ERROR("PoseSE3::update: Unsupported pose state representation!");
    }

    // Update the underlying state
    Eigen::Matrix4d T_new = Eigen::Matrix4d::Identity();
    T_new.block<3, 3>(0, 0) = C_ab_new;
    T_new.block<3, 1>(0, 3) = pos_new;
  }

  std::shared_ptr<ov_type::Type> clone() override {
    auto clone = std::shared_ptr<PoseSE3>(new PoseSE3(_pose_state_representation));
    clone->set_value(this->value());
    clone->set_fej(this->fej());
    return clone;
  }

  void fromMatrix(const Eigen::Matrix4d &T_new) {
    Eigen::Matrix3d C_ab = T_new.block<3, 3>(0, 0);
    Eigen::Vector3d pos = T_new.block<3, 1>(0, 3);

    Eigen::Quaterniond q_ab{C_ab};

    Eigen::Matrix<double, 7, 1> newX;
    newX(0, 0) = q_ab.x();
    newX(1, 0) = q_ab.y();
    newX(2, 0) = q_ab.z();
    newX(3, 0) = q_ab.w();
    newX.block<3, 1>(4, 0) = pos;

    set_value(newX);
  }

  Eigen::Matrix4d asMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 7, 1> value_mat = value();
    Eigen::Quaterniond q_ab{value_mat(0, 0), value_mat(1, 0), value_mat(2, 0), value_mat(3, 0)};
    Eigen::Matrix3d C_ab = q_ab.toRotationMatrix();
    T.block<3, 3>(0, 0) = C_ab;
    T.block<3, 1>(0, 3) = value_mat.block<3, 1>(4, 0);
    return T;
  }

  // Attitude and position accessors
  Eigen::Matrix3d Rot() const {
    Eigen::Matrix<double, 7, 1> value_mat = value();
    Eigen::Quaterniond q_ab{value_mat(0, 0), value_mat(1, 0), value_mat(2, 0), value_mat(3, 0)};
    return q_ab.toRotationMatrix();
  }

  Eigen::Matrix3d Rot_fej() const {
    Eigen::Matrix<double, 7, 1> fej_mat = fej();
    Eigen::Quaterniond q_ab{fej_mat(0, 0), fej_mat(1, 0), fej_mat(2, 0), fej_mat(3, 0)};
    return q_ab.toRotationMatrix();
  }

  Eigen::Matrix<double, 4, 1> quat() const {
    Eigen::Matrix<double, 7, 1> value_mat = value();
    return value_mat.head<4>();
  }

  Eigen::Matrix<double, 4, 1> quat_fej() const {
    Eigen::Matrix<double, 7, 1> fej_mat = fej();
    return fej_mat.head<4>();
  }

  Eigen::Vector3d pos() const {
    Eigen::Matrix<double, 7, 1> value_mat = value();
    return value_mat.tail<3>();
  }

  Eigen::Vector3d pos_fej() const {
    Eigen::Matrix<double, 7, 1> fej_mat = fej();
    return fej_mat.tail<3>();
  }

protected:
  PoseStateRepresentation _pose_state_representation;
};
