import React from 'react';
import { Button, Form, Input, Space, Typography } from 'antd';
import { useStore } from '../../store';

interface HFTokenInputProps {
  disabled?: boolean;
}

export const HFTokenInput: React.FC<HFTokenInputProps> = ({ disabled = false }) => {
  const { hfToken, setHfToken, clearHfToken } = useStore((state) => ({
    hfToken: state.hfToken,
    setHfToken: state.setHfToken,
    clearHfToken: state.clearHfToken,
  }));

  return (
    <Form.Item label="HF Token (Optional)">
      <Input.Password
        value={hfToken}
        onChange={(e) => setHfToken(e.target.value)}
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxx"
        disabled={disabled}
      />
      <Space style={{ marginTop: 6 }} size={8}>
        {hfToken && (
          <Button size="small" onClick={clearHfToken} disabled={disabled}>
            Clear
          </Button>
        )}
        <Typography.Text style={{ fontSize: 11, color: '#8c8c8c' }}>
          Token is kept in-memory for this session only.
        </Typography.Text>
      </Space>
      <Typography.Text style={{ fontSize: 11, color: '#8c8c8c', display: 'block', marginTop: 4 }}>
        Required for gated model downloads. Create one at{' '}
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
          huggingface.co/settings/tokens
        </a>
        .
      </Typography.Text>
    </Form.Item>
  );
};
